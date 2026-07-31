"""
Graph Memory Module
-------------------
Provides a lightweight Knowledge Graph using NetworkX.
Stores entities (Nodes) and relationships (Edges) to allow the agent to "connect the dots".

Data is persisted to `knowledge_graph.json`.
"""
import functools
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Any

import networkx as nx

from agent.utils import safe_print
from tools.exception_logger import log_exceptions
from tools.user_profile import get_active_profile, get_data_path


def _synchronized(method):
    """Serialize a GraphMemory operation so its reload→mutate→save is atomic.

    The graph is a process-wide singleton with a single in-memory copy, and
    _ensure_profile_sync() reloads it for whichever profile is currently active.
    Without serialization, a concurrent request for a different profile can swap
    the shared graph between one operation's reload and its save — writing one
    user's graph into another user's file (the cross-profile contamination the
    ContextVar fix does NOT cover, since this is shared mutable state, not a lost
    binding). A re-entrant lock (methods call one another) makes each public
    operation atomic with respect to the active profile.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class GraphMemory:
    def __init__(self):
        self._lock = threading.RLock()
        self._graph = nx.MultiDiGraph() # MultiDiGraph allows multiple edges between nodes
        self.current_profile = get_active_profile()
        self.on_save_callback = None # Hook for real-time sync (e.g. WebSockets)
        self.load()

    @property
    def graph(self):
        self._ensure_profile_sync()
        return self._graph

    @graph.setter
    def graph(self, value):
        self._graph = value

    def _ensure_profile_sync(self):
        """Check if the active profile changed, and reload if necessary.

        The reload runs under the lock (double-checked) so it cannot replace the
        shared graph while another operation is mid-flight. Inside a
        @_synchronized method the caller already holds the (re-entrant) lock."""
        active = get_active_profile()
        if self.current_profile != active:
            with self._lock:
                if self.current_profile != active:
                    self.current_profile = active
                    self.load()

    def load(self):
        """Load the graph from JSON file with cross-version compatibility."""
        graph_file = get_data_path("knowledge_graph.json")
        if os.path.exists(graph_file):
            try:
                # A concurrent writer can momentarily leave the file truncated.
                # Retry a few times on a decode error before giving up — a
                # transient race should never reset us to an empty graph.
                data = None
                for attempt in range(4):
                    try:
                        with open(graph_file) as f:
                            data = json.load(f)
                        break
                    except json.JSONDecodeError:
                        if attempt == 3:
                            raise
                        import time as _time
                        _time.sleep(0.1)

                # NetworkX has flipped between 'links' and 'edges' keys across versions.
                # We detect what's in the file and tell node_link_graph what to expect.
                edge_key = "links" if "links" in data else "edges"

                # (Newer NX versions use 'edges=' keyword, older used 'link=')
                try:
                    # Try modern 'edges' param (NX 3.2+)
                    self._graph = nx.node_link_graph(data, edges=edge_key)
                except TypeError:
                    try:
                        # Fallback for older NX 3.x which used 'link'
                        self._graph = nx.node_link_graph(data, link=edge_key)
                    except TypeError:
                        # Final fallback for very old or core defaults
                        self._graph = nx.node_link_graph(data)

                # Ensure it's a MultiDiGraph if the data says so
                if data.get("multigraph") and not isinstance(self._graph, nx.MultiDiGraph):
                    # Convert to MultiDiGraph if necessary
                    self._graph = nx.MultiDiGraph(self._graph)

            except Exception as e:
                safe_print(f"⚠️ Failed to load Knowledge Graph from {graph_file}: {e}")
                self._graph = nx.MultiDiGraph()
        else:
            self._graph = nx.MultiDiGraph()

    def save(self):
        """Save the graph to JSON file with standardized naming."""
        try:
            graph_file = get_data_path("knowledge_graph.json")
            # NetworkX changed the parameter name from 'link' to 'edges' in recent 3.x versions.
            # We use a robust fallback to support both old and modern environments.
            try:
                # Try new 'edges' param (NX 3.2+)
                data = nx.node_link_data(self._graph, edges="links")
            except TypeError:
                # Fallback to 'link' param (older NX 3.x)
                data = nx.node_link_data(self._graph, link="links")

            # Atomic write: serialize to a temp file in the same directory, then
            # os.replace() over the target. The rename is atomic, so concurrent
            # workers calling load() never observe a half-written (truncated) file
            # — they see either the complete old or complete new graph. A plain
            # open(graph_file, 'w') truncates first and races readers into a
            # JSONDecodeError (and a silent reset to an empty graph).
            import tempfile
            dir_name = os.path.dirname(graph_file) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".knowledge_graph.", suffix=".tmp", dir=dir_name)
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, graph_file)
            except BaseException:
                # Don't leave a stray temp file behind on any failure.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Trigger real-time sync if callback is registered
            if self.on_save_callback:
                try:
                    # We pass the data so the callback doesn't have to re-read it
                    self.on_save_callback(data)
                except Exception as cb_e:
                    safe_print(f"⚠️ Graph save callback failed: {cb_e}")

        except Exception as e:
            safe_print(f"⚠️ Failed to save Knowledge Graph: {e}")

    @_synchronized
    def delete_entity(self, name: str):
        """Remove a node and all connected edges from the graph."""
        self._ensure_profile_sync()
        name = name.strip()
        if self.graph.has_node(name):
            self.graph.remove_node(name)
            self.save()
            return True
        return False

    @_synchronized
    def delete_relationship(self, source: str, target: str, relation: str):
        """Remove a specific edge from the graph."""
        self._ensure_profile_sync()
        if self.graph.has_edge(source, target):
            # Iterate through all edges between u and v
            keys_to_remove = []
            for k, v in self.graph.get_edge_data(source, target).items():
                if v.get("relation") == relation:
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                self.graph.remove_edge(source, target, key=k)

            if keys_to_remove:
                self.save()
                return True
        return False

    @_synchronized
    def clear(self):
        """Reset the graph to empty state."""
        self._ensure_profile_sync()
        self.graph = nx.MultiDiGraph()
        self.save()

    # Hub node type enforcement — prevents "Unknown" on well-known nodes
    _HUB_TYPES = {
        "Portfolio": "UserPortfolio",
        "User": "Person",
    }

    @_synchronized
    def add_entity(self, name: str, type: str, attributes: dict[str, Any] = None):
        """
        Add a node (Entity) to the graph.
        e.g. add_entity("Google", "Company", {"sector": "Tech"})
        """
        self._ensure_profile_sync()
        name = name.strip()
        if not name: return

        # Enforce correct types for well-known hub nodes
        if name in self._HUB_TYPES and (type == "Unknown" or not type):
            type = self._HUB_TYPES[name]

        # Merge attributes if node exists
        if self.graph.has_node(name):
            current_attrs = self.graph.nodes[name]
            if attributes:
                current_attrs.update(attributes)
            # Ensure type is not overwritten with generic if specific exists,
            # but do upgrade old Unknown hub/entity records when a real type is known.
            if type != "Unknown" and current_attrs.get("type", "Unknown") == "Unknown":
                current_attrs["type"] = type
            self.graph.nodes[name].update(current_attrs)
        else:
            attrs = attributes or {}
            attrs["type"] = type
            self.graph.add_node(name, **attrs)

        self.save()

    @_synchronized
    def add_relationship(self, source: str, target: str, relation: str,
                         attributes: dict[str, Any] = None,
                         stale_after_days: int = None):
        """
        Add an edge (Relationship) between two entities.
        e.g. add_relationship("User", "Google", "WORKS_AT")

        Args:
            stale_after_days: If set, edge will be auto-pruned after this many days.
        """
        self._ensure_profile_sync()
        source = source.strip()
        target = target.strip()
        relation = relation.strip().upper()

        if not source or not target or not relation: return

        # Ensure nodes exist (auto-create as Unknown if missing)
        if not self.graph.has_node(source):
            self.add_entity(source, "Unknown")
        if not self.graph.has_node(target):
            self.add_entity(target, "Unknown")

        # Check if edge already exists to avoid duplicates
        exists = False
        if self.graph.has_edge(source, target):
            for k, v in self.graph.get_edge_data(source, target).items():
                if v.get("relation") == relation:
                    exists = True
                    break

        if exists:
            # Update attributes of existing edge
            for k, v in self.graph.get_edge_data(source, target).items():
                if v.get("relation") == relation:
                    if attributes:
                        self.graph.edges[source, target, k].update(attributes)
                    if stale_after_days is not None:
                        self.graph.edges[source, target, k]["stale_after_days"] = stale_after_days
                    # Refresh timestamp on update
                    self.graph.edges[source, target, k]["updated_at"] = datetime.now().isoformat()
                    break
        else:
            attrs = attributes or {}
            attrs["relation"] = relation
            attrs["created_at"] = datetime.now().isoformat()
            if stale_after_days is not None:
                attrs["stale_after_days"] = stale_after_days
            self.graph.add_edge(source, target, **attrs)

        self.save()

    @_synchronized
    def prune_stale(self) -> int:
        """
        Remove edges that have exceeded their stale_after_days threshold.
        Returns the number of edges removed.
        """
        self._ensure_profile_sync()
        now = datetime.now()
        edges_to_remove = []

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            stale_days = data.get("stale_after_days")
            created_at = data.get("created_at")
            if stale_days and created_at:
                try:
                    created = datetime.fromisoformat(created_at)
                    if (now - created) > timedelta(days=stale_days):
                        edges_to_remove.append((u, v, k))
                except (ValueError, TypeError):
                    continue

        for u, v, k in edges_to_remove:
            self.graph.remove_edge(u, v, key=k)

        if edges_to_remove:
            safe_print(f"🕰️ Pruned {len(edges_to_remove)} stale edges from Knowledge Graph")
            self.save()

        return len(edges_to_remove)

    @_synchronized
    def prune_orphans(self) -> int:
        """
        Remove nodes that have no edges, no 'owned' flag, and type is 'Unknown' or 'Theme'.
        Also calls prune_stale() first to expire timed-out edges.
        Returns the number of nodes removed.
        """
        self._ensure_profile_sync()

        # First, expire stale edges
        self.prune_stale()

        orphans = []
        KEEP_TYPES = {"Sector", "UserPortfolio", "Person", "Broker"}
        for node, data in self.graph.nodes(data=True):
            is_owned = data.get("owned", False)
            is_protected_type = data.get("type") in KEEP_TYPES
            has_edges = self.graph.degree(node) > 0
            if not is_owned and not is_protected_type and not has_edges:
                orphans.append(node)

        for o in orphans:
            self.graph.remove_node(o)

        if orphans:
            self.save()
            safe_print(f"🧹 Pruned {len(orphans)} orphan nodes from Knowledge Graph: {', '.join(orphans)}")

        return len(orphans)

    @_synchronized
    def get_context(self, entities: list[str], depth: int = 1) -> str:
        """
        Search the graph for the given entities and return a text summary of their neighbors.
        Useful for injecting into LLM context.
        """
        self._ensure_profile_sync()
        self.prune_stale()
        found_entities = [e for e in entities if self.graph.has_node(e)]
        if not found_entities:
            return ""

        context_lines = []
        visited = set()

        for entity in found_entities:
            # Get neighbors (successors and predecessors)
            # Use ego_graph to get the subgraph within specific radius
            try:
                subgraph = nx.ego_graph(self.graph, entity, radius=depth)

                # Format relationships
                edges = subgraph.edges(data=True)
                for u, v, data in edges:
                    rel = data.get("relation", "RELATED_TO")

                    # Store as "Entity1 --RELATION--> Entity2"
                    line = f"- {u} --{rel}--> {v}"

                    # Add node types if known
                    u_type = self.graph.nodes[u].get("type", "")
                    v_type = self.graph.nodes[v].get("type", "")
                    if u_type and u_type != "Unknown":
                        line += f" ({u_type})"
                    if v_type and v_type != "Unknown":
                        line += f" ({v_type})"

                    if line not in visited:
                        context_lines.append(line)
                        visited.add(line)

            except Exception as e:
                safe_print(f"Graph search error for {entity}: {e}")
                continue

        if not context_lines:
            return ""

        return "=== KNOWLEDGE GRAPH CONTEXT ===\n" + "\n".join(context_lines) + "\n===============================\n"

    @_synchronized
    def add_portfolio_context(self, holdings: list, sector_exposure: dict = None, correlations: list = None):
        """
        Auto-populate the graph with useful portfolio data.

        Args:
            holdings: List of holding dicts with 'symbol' and 'sector' keys
            sector_exposure: Dict of {sector: percentage}
            correlations: List of tuples (symbol1, symbol2, correlation_value)
        """
        self._ensure_profile_sync()
        try:
            # 0. Clear 'owned' flag from all existing stocks first to ensure accuracy
            for node, data in self.graph.nodes(data=True):
                if data.get("type") == "Stock" or data.get("type") == "Ticker":
                    self.graph.nodes[node]["owned"] = False

            # 1. Add stock -> sector relationships and set ownership
            for h in holdings:
                if isinstance(h, dict):
                    symbol = h.get("symbol", "").upper()
                    sector = h.get("sector", "Unknown")
                    if symbol:
                        # Set owned attribute for visual highlighting in UI
                        self.add_entity(symbol, "Stock", {"sector": sector, "owned": True})
                        if sector and sector != "Unknown":
                            self.add_entity(sector, "Sector")
                            self.add_relationship(symbol, sector, "IN_SECTOR")

            # 2. Add sector exposure
            if sector_exposure:
                self.add_entity("Portfolio", "UserPortfolio")

                # Clear old exposure edges first to remove stale data (e.g. "Unknown" sector)
                if self.graph.has_node("Portfolio"):
                    edges_to_remove = []
                    for u, v, k, d in self.graph.out_edges("Portfolio", keys=True, data=True):
                        if d.get("relation") == "EXPOSED_TO":
                            edges_to_remove.append((u, v, k))
                    for u, v, k in edges_to_remove:
                        self.graph.remove_edge(u, v, key=k)

                for sector, pct in sector_exposure.items():
                    if sector and pct:
                        self.add_entity(sector, "Sector")
                        self.add_relationship("Portfolio", sector, "EXPOSED_TO",
                                            {"percentage": f"{pct:.1f}%"})

            # 3. Add top correlations (stocks that move together)
            if correlations:
                for sym1, sym2, corr in correlations[:5]:  # Top 5 only
                    if corr > 0.7:  # Only strong correlations
                        # Type the endpoints as Stock first. Correlation pairs are
                        # portfolio holdings, so without this they'd be auto-created
                        # as "Unknown" by add_relationship and then filtered out of
                        # the graph view (which hides Unknown nodes).
                        s1, s2 = sym1.strip().upper(), sym2.strip().upper()
                        self.add_entity(s1, "Stock")
                        self.add_entity(s2, "Stock")
                        self.add_relationship(s1, s2, "CORRELATED_WITH",
                                            {"strength": f"{corr:.2f}"})

            self.save()
            return True
        except Exception as e:
            safe_print(f"⚠️ Failed to add portfolio context: {e}")
            return False

    @_synchronized
    def get_portfolio_summary(self) -> str:
        """Get a concise summary of portfolio-related graph data."""
        self._ensure_profile_sync()
        self.prune_stale()
        sectors = []
        correlations = []
        interests = []

        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "")
            if rel == "EXPOSED_TO":
                pct = data.get("percentage", "")
                sectors.append(f"{v}: {pct}")
            elif rel == "CORRELATED_WITH":
                strength = data.get("strength", "")
                correlations.append(f"{u}↔{v} ({strength})")
            elif rel == "INTERESTED_IN" and u == "User":
                interests.append(v)

        lines = []
        if sectors:
            lines.append("📊 Sector Exposure: " + ", ".join(sectors[:5]))
        if interests:
            lines.append("👀 Tracking: " + ", ".join(interests[:10]))  # Top 10 tracked
        if correlations:
            lines.append("🔗 Correlated Pairs: " + ", ".join(correlations[:3]))

        return "\n".join(lines) if lines else "No portfolio context stored"

# Singleton instance
graph_memory = GraphMemory()

@log_exceptions()
def add_memory_fragment(source: str, relation: str, target: str):
    """Helper for the agent to quickly add a triplet."""
    graph_memory.add_relationship(source, target, relation)

@log_exceptions()
def search_graph(query: str):
    """Helper to search the graph by a single entity name."""
    return graph_memory.get_context([query])

if __name__ == "__main__":
    # Test
    gm = GraphMemory()
    gm.add_entity("User", "Person")
    gm.add_entity("Google", "Company", {"sector": "Tech"})
    gm.add_relationship("User", "Google", "WORKS_AT")
    gm.add_relationship("Google", "NVDA", "PARTNER_WITH")

    print(gm.get_context(["User", "Google"], depth=2))
