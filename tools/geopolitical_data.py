"""
Geopolitical supply-chain knowledge base (static data).

Extracted verbatim from geopolitical_scanner.py. These maps are STATIC
(not LLM-generated) for speed and reliability: the LLM interprets news,
this knowledge base maps countries/commodities to tickers.
"""

from typing import Any

# =============================================================================
# SUPPLY CHAIN KNOWLEDGE BASE
# =============================================================================

# Country/Region → List of (commodity, global_share_pct, description)
COUNTRY_COMMODITY_MAP: dict[str, list[dict[str, Any]]] = {
    # --- Middle East ---
    "qatar": [
        {"commodity": "helium", "share": 30, "desc": "~30% of global helium supply"},
        {"commodity": "lng", "share": 25, "desc": "World's largest LNG exporter"},
        {"commodity": "natural_gas", "share": 12, "desc": "Major natural gas producer"},
    ],
    "saudi_arabia": [
        {"commodity": "oil", "share": 12, "desc": "Top-3 global oil producer"},
        {"commodity": "petrochemicals", "share": 8, "desc": "Major petrochemical exporter"},
    ],
    "iran": [
        {"commodity": "oil", "share": 4, "desc": "Significant crude exporter (when unsanctioned)"},
        {"commodity": "natural_gas", "share": 6, "desc": "2nd largest gas reserves globally"},
    ],
    "iraq": [
        {"commodity": "oil", "share": 5, "desc": "5th largest oil producer"},
    ],
    "uae": [
        {"commodity": "oil", "share": 4, "desc": "Significant oil producer"},
        {"commodity": "aluminum", "share": 5, "desc": "Major aluminum smelter (EGA)"},
    ],
    "middle_east": [
        {"commodity": "oil", "share": 33, "desc": "~33% of global oil production"},
        {"commodity": "natural_gas", "share": 20, "desc": "~20% of global gas production"},
    ],

    # --- Europe / Russia ---
    "russia": [
        {"commodity": "palladium", "share": 40, "desc": "~40% of global palladium"},
        {"commodity": "wheat", "share": 18, "desc": "Top wheat exporter"},
        {"commodity": "oil", "share": 12, "desc": "Top-3 oil producer"},
        {"commodity": "natural_gas", "share": 17, "desc": "Major pipeline gas supplier to Europe"},
        {"commodity": "nickel", "share": 10, "desc": "Major nickel producer (Nornickel)"},
        {"commodity": "fertilizer", "share": 15, "desc": "Major potash/nitrogen exporter"},
        {"commodity": "titanium", "share": 20, "desc": "Key titanium sponge supplier (aerospace)"},
    ],
    "ukraine": [
        {"commodity": "wheat", "share": 10, "desc": "Major wheat/corn exporter"},
        {"commodity": "neon_gas", "share": 50, "desc": "~50% of semiconductor-grade neon"},
        {"commodity": "sunflower_oil", "share": 50, "desc": "~50% of global sunflower oil"},
    ],

    # --- Asia ---
    "taiwan": [
        {"commodity": "semiconductors", "share": 65, "desc": "~65% of global advanced chip manufacturing (TSMC)"},
    ],
    "china": [
        {"commodity": "rare_earths", "share": 70, "desc": "~70% of global rare earth mining"},
        {"commodity": "solar_panels", "share": 80, "desc": "~80% of global solar panel production"},
        {"commodity": "ev_batteries", "share": 60, "desc": "~60% of global EV battery production"},
        {"commodity": "gallium", "share": 80, "desc": "~80% of gallium (semiconductor material)"},
        {"commodity": "germanium", "share": 60, "desc": "~60% of germanium (fiber optics, military)"},
        {"commodity": "steel", "share": 55, "desc": "~55% of global steel production"},
    ],
    "japan": [
        {"commodity": "auto_parts", "share": 15, "desc": "Major auto component supplier"},
        {"commodity": "semiconductors", "share": 10, "desc": "Key chip equipment/materials"},
    ],
    "south_korea": [
        {"commodity": "memory_chips", "share": 60, "desc": "~60% of global DRAM/NAND (Samsung, SK Hynix)"},
        {"commodity": "batteries", "share": 20, "desc": "Major battery cell manufacturer"},
        {"commodity": "shipbuilding", "share": 35, "desc": "~35% of global shipbuilding"},
    ],
    "india": [
        {"commodity": "pharmaceuticals", "share": 20, "desc": "~20% of global generic drugs"},
        {"commodity": "rice", "share": 40, "desc": "~40% of global rice exports"},
        {"commodity": "it_services", "share": 15, "desc": "Major IT outsourcing hub"},
    ],

    # --- Americas ---
    "chile": [
        {"commodity": "copper", "share": 28, "desc": "~28% of global copper mining"},
        {"commodity": "lithium", "share": 25, "desc": "~25% of global lithium production"},
    ],
    "argentina": [
        {"commodity": "lithium", "share": 10, "desc": "Growing lithium producer (Lithium Triangle)"},
        {"commodity": "soy", "share": 15, "desc": "Major soy/agricultural exporter"},
    ],
    "brazil": [
        {"commodity": "iron_ore", "share": 20, "desc": "~20% of global iron ore"},
        {"commodity": "soy", "share": 30, "desc": "~30% of global soy exports"},
        {"commodity": "coffee", "share": 35, "desc": "~35% of global coffee production"},
    ],
    "mexico": [
        {"commodity": "silver", "share": 23, "desc": "~23% of global silver mining"},
        {"commodity": "auto_manufacturing", "share": 8, "desc": "Major auto assembly hub (nearshoring)"},
    ],
    "canada": [
        {"commodity": "potash", "share": 30, "desc": "~30% of global potash (fertilizer)"},
        {"commodity": "oil_sands", "share": 4, "desc": "~4% of global oil (oil sands)"},
        {"commodity": "uranium", "share": 15, "desc": "~15% of global uranium"},
        {"commodity": "lumber", "share": 12, "desc": "Major softwood lumber exporter"},
    ],

    # --- Africa ---
    "congo": [
        {"commodity": "cobalt", "share": 70, "desc": "~70% of global cobalt (EV batteries)"},
        {"commodity": "copper", "share": 5, "desc": "Growing copper producer"},
    ],
    "south_africa": [
        {"commodity": "platinum", "share": 70, "desc": "~70% of global platinum"},
        {"commodity": "palladium", "share": 35, "desc": "~35% of global palladium"},
        {"commodity": "chrome", "share": 40, "desc": "~40% of global chrome ore"},
        {"commodity": "manganese", "share": 30, "desc": "~30% of global manganese"},
    ],
    "nigeria": [
        {"commodity": "oil", "share": 3, "desc": "Africa's largest oil producer"},
    ],
    "morocco": [
        {"commodity": "phosphate", "share": 30, "desc": "~30% of global phosphate (fertilizer)"},
    ],

    # --- Oceania ---
    "australia": [
        {"commodity": "iron_ore", "share": 55, "desc": "~55% of global iron ore exports"},
        {"commodity": "coal", "share": 25, "desc": "~25% of global coal exports"},
        {"commodity": "lng", "share": 20, "desc": "Major LNG exporter"},
        {"commodity": "lithium", "share": 50, "desc": "~50% of global lithium mining"},
    ],
    "indonesia": [
        {"commodity": "nickel", "share": 35, "desc": "~35% of global nickel"},
        {"commodity": "palm_oil", "share": 55, "desc": "~55% of global palm oil"},
        {"commodity": "tin", "share": 20, "desc": "~20% of global tin"},
        {"commodity": "coal", "share": 15, "desc": "Major thermal coal exporter"},
    ],

    # --- Strategic Chokepoints ---
    "panama_canal": [
        {"commodity": "shipping", "share": 5, "desc": "~5% of global trade transits here"},
    ],
    "suez_canal": [
        {"commodity": "shipping", "share": 12, "desc": "~12% of global trade transits here"},
        {"commodity": "oil", "share": 9, "desc": "~9% of global oil transits here"},
    ],
    "strait_of_hormuz": [
        {"commodity": "oil", "share": 21, "desc": "~21% of global oil transits here"},
        {"commodity": "lng", "share": 25, "desc": "~25% of global LNG transits here"},
    ],
    "strait_of_malacca": [
        {"commodity": "shipping", "share": 25, "desc": "~25% of global trade transits here"},
        {"commodity": "oil", "share": 16, "desc": "~16% of global oil transits here"},
    ],
}

# Commodity → Beneficiary tickers
# These are companies that BENEFIT when supply is disrupted
# (producers, substitutes, shipping, or downstream with pricing power)
COMMODITY_TICKER_MAP: dict[str, dict[str, Any]] = {
    # Energy
    "oil": {
        "producers": ["XOM", "CVX", "COP", "EOG", "PXD", "DVN", "SU.TO", "CNQ.TO"],
        "services": ["SLB", "HAL", "BKR"],
        "etfs": ["USO", "XLE", "OIH"],
        "desc": "Crude oil producers & services",
    },
    "natural_gas": {
        "producers": ["EQT", "AR", "RRC", "SWN", "CTRA"],
        "lng_players": ["LNG", "GLNG", "TELL", "NFE"],
        "etfs": ["UNG", "KOLD", "BOIL"],
        "desc": "Natural gas producers & LNG exporters",
    },
    "lng": {
        "producers": ["LNG", "GLNG", "TELL", "NFE"],
        "carriers": ["FLNG", "KNOP"],
        "infrastructure": ["KMI", "WMB", "ET"],
        "etfs": ["UNG"],
        "desc": "LNG producers, carriers, and infrastructure",
    },
    "helium": {
        "related": ["LNG", "GLNG", "AR", "EQT", "APD", "LIN"],
        "desc": "No pure helium stocks — tied to LNG/gas producers + industrial gas (APD, LIN)",
    },
    "coal": {
        "producers": ["BTU", "ARCH", "CEIX", "HCC", "ARLP"],
        "desc": "Coal mining companies",
    },
    "oil_sands": {
        "producers": ["SU.TO", "CNQ.TO", "CVE.TO", "IMO.TO"],
        "desc": "Canadian oil sands producers",
    },

    # Metals & Mining
    "copper": {
        "miners": ["FCX", "SCCO", "TECK", "HBM.TO"],
        "etfs": ["COPX", "CPER"],
        "desc": "Copper miners benefit from supply cuts",
    },
    "lithium": {
        "miners": ["ALB", "SQM", "LAC", "LTHM", "PLL"],
        "etfs": ["LIT"],
        "desc": "Lithium producers for EV batteries",
    },
    "cobalt": {
        "miners": ["CMCL", "GLEN.L"],
        "battery_makers": ["ALB", "LTHM"],
        "desc": "Cobalt for EV batteries, mostly from Congo",
    },
    "nickel": {
        "miners": ["VALE", "TECK", "BHP"],
        "desc": "Nickel for stainless steel & EV batteries",
    },
    "palladium": {
        "miners": ["SBSW", "IMPUY"],
        "substitutes": ["platinum → PPLT"],
        "desc": "Palladium for catalytic converters — Russia/SA supply",
    },
    "platinum": {
        "miners": ["SBSW", "IMPUY", "ANGPY"],
        "etfs": ["PPLT"],
        "desc": "Platinum mining, mostly South Africa",
    },
    "iron_ore": {
        "miners": ["BHP", "RIO", "VALE", "CLF"],
        "desc": "Iron ore for steelmaking",
    },
    "silver": {
        "miners": ["AG", "PAAS", "MAG", "HL"],
        "etfs": ["SLV"],
        "desc": "Silver mining companies",
    },
    "rare_earths": {
        "miners": ["MP", "UUUU"],
        "etfs": ["REMX"],
        "desc": "Rare earths — 70% from China, critical for EVs/defense",
    },
    "aluminum": {
        "producers": ["AA", "CENX"],
        "desc": "Aluminum smelters",
    },
    "titanium": {
        "users": ["BA", "LMT", "RTX"],
        "producers": ["TIE"],
        "desc": "Titanium critical for aerospace — Russia a key supplier",
    },
    "manganese": {
        "miners": ["BHP", "VALE"],
        "desc": "Manganese for steel & batteries",
    },
    "chrome": {
        "miners": ["GLEN.L"],
        "desc": "Chrome for stainless steel",
    },
    "uranium": {
        "miners": ["CCJ", "UEC", "DNN", "NXE", "URG"],
        "etfs": ["URNM", "URA"],
        "desc": "Uranium for nuclear power",
    },

    # Agriculture
    "wheat": {
        "agri": ["ADM", "BG", "INGR"],
        "fertilizer": ["MOS", "NTR", "CF"],
        "desc": "Grain disruption benefits exporters & fertilizer cos",
    },
    "soy": {
        "agri": ["ADM", "BG", "CTVA"],
        "desc": "Soy/grain trading companies",
    },
    "coffee": {
        "roasters": ["SBUX", "KDP"],
        "traders": ["ADM"],
        "desc": "Coffee supply disruption",
    },
    "rice": {
        "agri": ["ADM", "BG"],
        "desc": "Rice export disruption",
    },
    "fertilizer": {
        "producers": ["MOS", "NTR", "CF", "IPI"],
        "desc": "Fertilizer (potash, nitrogen, phosphate)",
    },
    "potash": {
        "producers": ["NTR", "MOS", "IPI"],
        "desc": "Potash fertilizer",
    },
    "phosphate": {
        "producers": ["MOS", "IPI"],
        "desc": "Phosphate fertilizer",
    },
    "palm_oil": {
        "producers": ["INDO"],
        "substitutes": ["ADM", "BG"],
        "desc": "Palm oil — Indonesia dominates",
    },
    "sunflower_oil": {
        "alternatives": ["ADM", "BG"],
        "desc": "Sunflower oil — substitutes benefit",
    },
    "lumber": {
        "producers": ["WY", "RYN", "PCH"],
        "desc": "Softwood lumber",
    },

    # Technology
    "semiconductors": {
        "foundries": ["TSM", "INTC", "GFS", "UMC"],
        "equipment": ["ASML", "AMAT", "LRCX", "KLAC", "TER"],
        "design": ["NVDA", "AMD", "AVGO", "QCOM"],
        "desc": "Chip supply disruption (mainly Taiwan/China tension)",
    },
    "memory_chips": {
        "producers": ["MU", "WDC", "STX"],
        "desc": "DRAM/NAND memory — Samsung/SK Hynix dominant",
    },
    "neon_gas": {
        "beneficiaries": ["ASML", "AMAT", "LRCX"],
        "desc": "Neon for chip lithography — Ukraine major supplier",
    },
    "solar_panels": {
        "us_alternatives": ["FSLR", "RUN", "ENPH", "SEDG"],
        "desc": "Solar manufacturing alternatives to China",
    },
    "ev_batteries": {
        "producers": ["TSLA", "ALB", "SQM", "LTHM", "QS"],
        "etfs": ["LIT"],
        "desc": "EV battery supply chain",
    },
    "gallium": {
        "users": ["INTC", "TSM", "NVDA"],
        "desc": "Gallium for chip manufacturing — China controls 80%",
    },
    "germanium": {
        "users": ["II-VI (COHR)", "LITE"],
        "desc": "Germanium for fiber optics & military optics",
    },

    # Shipping & Trade
    "shipping": {
        "carriers": ["ZIM", "GOGL", "DAC", "SBLK", "MATX"],
        "container": ["ZIM", "MATX"],
        "dry_bulk": ["GOGL", "SBLK", "GNK"],
        "tankers": ["STNG", "TNK", "FRO", "EURN"],
        "desc": "Shipping disruption (canal closures, war zones)",
    },

    # Industrial
    "petrochemicals": {
        "producers": ["LYB", "DOW", "CE"],
        "desc": "Petrochemical producers",
    },
    "auto_parts": {
        "suppliers": ["BWA", "ALV", "APTV", "LEA"],
        "desc": "Auto parts supply chain disruption",
    },
    "auto_manufacturing": {
        "oems": ["GM", "F", "TM", "STLA"],
        "desc": "Auto manufacturing nearshoring beneficiaries",
    },
    "batteries": {
        "producers": ["ALB", "SQM", "LTHM", "QS", "PCRFY"],
        "etfs": ["LIT"],
        "desc": "Battery cell makers",
    },
    "shipbuilding": {
        "builders": ["HII", "GD"],
        "desc": "Naval / commercial shipbuilding",
    },
    "steel": {
        "producers": ["NUE", "STLD", "CLF", "X"],
        "desc": "Steel producers",
    },

    # Pharma & Other
    "pharmaceuticals": {
        "generic_alt": ["TEVA", "MYL", "CI"],
        "desc": "Generic drug alternatives if India supply disrupted",
    },
    "it_services": {
        "providers": ["INFY", "WIT", "CTSH", "ACN"],
        "desc": "IT outsourcing companies",
    },

    # Defense (triggered by military conflicts)
    "defense": {
        "contractors": ["LMT", "RTX", "NOC", "GD", "BA", "LHX"],
        "desc": "Defense contractors — benefit from military conflicts",
    },
}

DOWNSTREAM_EFFECTS_MAP = {
    "semiconductors": {
        "vulnerable_sectors": ["Auto Manufacturers", "Consumer Electronics", "Cloud Providers"],
        "bearish_tickers": ["GM", "F", "AAPL", "DELL", "HPQ"],
        "thesis": "Chip shortages severely compress margins and halt production for downstream hardware and automotive companies."
    },
    "oil": {
        "vulnerable_sectors": ["Airlines", "Logistics", "Consumer Discretionary"],
        "bearish_tickers": ["DAL", "UAL", "FDX", "UPS", "CCL"],
        "thesis": "Higher crude prices destroy operating margins for transport and reduce consumer discretionary spending."
    },
    "natural_gas": {
        "vulnerable_sectors": ["European Industrials", "Chemicals"],
        "bearish_tickers": ["DOW", "LYB", "BASFY"],
        "thesis": "Nat gas spikes act as a massive tax on heavy industry and chemicals."
    },
    "copper": {
        "vulnerable_sectors": ["Homebuilders", "EV Manufacturers"],
        "bearish_tickers": ["DHI", "LEN", "TSLA", "RIVN"],
        "thesis": "Copper shortages increase raw material costs for electrification and construction."
    },
    "shipping": {
        "vulnerable_sectors": ["Retailers", "E-commerce"],
        "bearish_tickers": ["WMT", "TGT", "AMZN", "NKE"],
        "thesis": "Freight spikes compress retail gross margins and cause inventory shortages."
    }
}

# Event type keywords → additional categories to scan
EVENT_TYPE_MAP = {
    "military": ["defense", "oil", "shipping"],
    "war": ["defense", "oil", "shipping"],
    "strike": ["defense", "oil"],
    "invasion": ["defense", "oil", "wheat", "shipping"],
    "sanctions": ["oil", "natural_gas", "shipping"],
    "embargo": ["oil", "natural_gas"],
    "blockade": ["shipping", "oil"],
    "earthquake": ["shipping", "steel"],
    "hurricane": ["oil", "natural_gas", "shipping"],
    "typhoon": ["shipping", "semiconductors"],
    "flood": ["wheat", "rice", "soy"],
    "drought": ["wheat", "coffee", "soy", "rice"],
    "trade_war": ["semiconductors", "rare_earths", "steel"],
    "tariff": ["steel", "aluminum", "auto_parts"],
    "export_ban": ["rare_earths", "semiconductors", "fertilizer"],
    "nationalization": ["lithium", "copper", "oil"],
    "coup": ["oil", "cobalt", "copper"],
    "revolution": ["oil", "shipping"],
    "pandemic": ["pharmaceuticals", "shipping", "semiconductors"],
    "nuclear": ["uranium", "defense"],
}

# Country aliases for fuzzy matching
COUNTRY_ALIASES = {
    "drc": "congo",
    "democratic republic of congo": "congo",
    "rsa": "south_africa",
    "south korea": "south_korea",
    "s. korea": "south_korea",
    "rok": "south_korea",
    "prc": "china",
    "peoples republic": "china",
    "saudi": "saudi_arabia",
    "ksa": "saudi_arabia",
    "emirates": "uae",
    "united arab emirates": "uae",
    "hormuz": "strait_of_hormuz",
    "malacca": "strait_of_malacca",
    "suez": "suez_canal",
    "panama": "panama_canal",
    "persian gulf": "middle_east",
    "gulf": "middle_east",
    "mideast": "middle_east",
    "taiwan strait": "taiwan",
    "formosa": "taiwan",
}

# Historical conflict-era peak prices for key commodities
# Used to calculate "conflict premium" — how much MORE a commodity can spike
# Key insight: markets ALWAYS underestimate the duration of military conflicts
CONFLICT_PEAK_PRICES: dict[str, dict[str, Any]] = {
    "oil": {
        # CL=F (WTI Crude)
        "ticker": "CL=F",
        "peaks": [
            {"event": "Russia-Ukraine 2022", "peak": 130, "note": "Oil hit $130 on supply fears"},
            {"event": "Iran tensions 2019-2020", "peak": 97, "note": "Soleimani strike / Saudi Aramco drone"},
            {"event": "Libya civil war 2011", "peak": 113, "note": "Arab Spring supply disruption"},
            {"event": "Iraq War 2008", "peak": 147, "note": "All-time peak during Iraq war + financial crisis"},
        ],
        "historical_avg_conflict_premium_pct": 40,  # Oil typically spikes 40% in major conflicts
    },
    "natural_gas": {
        "ticker": "NG=F",
        "peaks": [
            {"event": "Russia-Ukraine / EU gas crisis 2022", "peak": 10.0, "note": "US nat gas hit $10/MMBtu"},
            {"event": "Winter storm 2021", "peak": 6.5, "note": "Texas freeze supply shock"},
        ],
        "historical_avg_conflict_premium_pct": 60,
    },
    "wheat": {
        "ticker": "ZW=F",
        "peaks": [
            {"event": "Russia-Ukraine 2022", "peak": 1350, "note": "Wheat hit $13.50/bu on Ukraine export block"},
            {"event": "Russian export ban 2010", "peak": 870, "note": "Drought + export ban"},
        ],
        "historical_avg_conflict_premium_pct": 50,
    },
    "palladium": {
        "ticker": "PA=F",
        "peaks": [
            {"event": "Russia-Ukraine 2022", "peak": 3440, "note": "Russia = 40% of global supply"},
        ],
        "historical_avg_conflict_premium_pct": 80,
    },
    "nickel": {
        "ticker": "^SPGSN",  # No great futures ticker, use index as proxy
        "peaks": [
            {"event": "Russia-Ukraine / LME squeeze 2022", "peak": 100000, "note": "LME nickel doubled in a day"},
        ],
        "historical_avg_conflict_premium_pct": 60,
    },
    "lng": {
        "ticker": "LNG",  # Use Cheniere as LNG proxy
        "peaks": [
            {"event": "EU gas crisis 2022", "peak": 190, "note": "Cheniere hit $190 on LNG demand"},
        ],
        "historical_avg_conflict_premium_pct": 50,
    },
    "shipping": {
        "ticker": "ZIM",  # ZIM as container shipping proxy
        "peaks": [
            {"event": "Houthi Red Sea attacks 2024", "peak": 25, "note": "Shipping rates spiked"},
            {"event": "Supply chain crisis 2021", "peak": 90, "note": "COVID shipping boom"},
        ],
        "historical_avg_conflict_premium_pct": 100,
    },
    "defense": {
        "ticker": "LMT",  # Lockheed as defense proxy
        "peaks": [
            {"event": "Russia-Ukraine 2022-23", "peak": 500, "note": "Defense spending surge"},
        ],
        "historical_avg_conflict_premium_pct": 30,
    },
    "helium": {
        "ticker": "APD",  # Air Products as helium proxy
        "peaks": [
            {"event": "Qatar blockade 2017", "peak": 215, "note": "Helium shortage alert"},
        ],
        "historical_avg_conflict_premium_pct": 25,
    },
    "copper": {
        "ticker": "HG=F",
        "peaks": [
            {"event": "Post-COVID stimulus 2021-22", "peak": 5.0, "note": "Copper hit $5/lb"},
        ],
        "historical_avg_conflict_premium_pct": 30,
    },
    "uranium": {
        "ticker": "CCJ",  # Cameco as uranium proxy
        "peaks": [
            {"event": "Nuclear renaissance 2024", "peak": 65, "note": "AI power demand + nuclear"},
        ],
        "historical_avg_conflict_premium_pct": 40,
    },
    "lithium": {
        "ticker": "ALB",  # Albemarle as lithium proxy
        "peaks": [
            {"event": "EV boom 2022", "peak": 330, "note": "Lithium mania"},
        ],
        "historical_avg_conflict_premium_pct": 50,
    },
    "fertilizer": {
        "ticker": "MOS",
        "peaks": [
            {"event": "Russia-Ukraine / Belarus sanctions 2022", "peak": 79, "note": "Potash/fertilizer crisis"},
        ],
        "historical_avg_conflict_premium_pct": 60,
    },
}
