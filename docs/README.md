# Documentation Index

Complete documentation for CairnIQ.

## 📚 User Documentation

- **[Installation Guide](user-guide/INSTALLATION.md)** — installer walkthrough, Guided Setup Wizard, LLM/data provider comparison, portfolio setup, launch options, fallback architecture
- **[User Guide](user-guide/USER_GUIDE.md)** — chat interface, agent routing, dashboard, portfolio management, memory system, thesis journal, tips & best practices
- **[Troubleshooting Guide](user-guide/TROUBLESHOOTING.md)** — install / startup / API / performance / data / tool issues

## 🔧 Technical Documentation

- **[Project Structure](PROJECT_STRUCTURE.md)** — directory layout, core components, data flow, dependencies
- **[Architecture](technical/ARCHITECTURE.md)** — runtime shape, agent flow, operational constraints
- **[API Reference](technical/API.md)** — every REST/SSE endpoint with request/response shapes
- **[Adding Tools](technical/ADDING_TOOLS.md)** — extending the agent's capabilities
- **[Tool Capabilities](technical/TOOL_CAPABILITIES.md)** — inventory of available tools and their inputs
- **[Funnel Configuration Guide](technical/FUNNEL_CONFIG.md)** — tuning the opportunity scanner via `user_data/funnel_config.json` (field reference + recipes)

## 🚀 Operations

- **[Launcher Modes](LAUNCHER_MODES.md)** — production (`CairnIQ.command`) vs demo (`start_demo.sh`), feature isolation matrix
- **[Changelog](CHANGELOG.md)** — version history and upgrade notes

## 📁 Layout

```
docs/
├── README.md                    # This index
├── user-guide/
│   ├── INSTALLATION.md          # Setup guide
│   ├── USER_GUIDE.md            # Usage manual
│   └── TROUBLESHOOTING.md       # Problem solutions
├── technical/
│   ├── ADDING_TOOLS.md          # Guide to adding new tools
│   ├── API.md                   # REST API reference
│   ├── ARCHITECTURE.md          # System architecture
│   ├── TOOL_CAPABILITIES.md     # Tool inventory and capabilities
│   └── FUNNEL_CONFIG.md         # Opportunity scanner config guide
├── archive/                     # Internal debug / fix notes (not distributed)
├── PROJECT_STRUCTURE.md         # Codebase overview
├── LAUNCHER_MODES.md            # Launch modes reference
└── CHANGELOG.md                 # Version history
```

## 📞 Support

If something isn't covered here:
1. Check the [Troubleshooting Guide](user-guide/TROUBLESHOOTING.md)
2. Review structured logs in `logs/` (server, tools, frontend, chat_runtime, agent)
3. Run an in-app diagnostic — just type `Run a full system health check` in the chat
4. Open an issue on GitHub (security issues: use the private vulnerability reporting flow per [SECURITY.md](../SECURITY.md))
