PhoneCompareBot — Multi-Agent Smartphone Comparison Bot

A multi-agent bot that automates searching and comparing smartphone specs across the web. Useful for getting side-by-side comparisons quickly, without manual copying and pasting from sites.

🚀 Features

Automatically retrieve smartphone specifications by crawling / scraping browser data

Compare multiple phones side by side

Modular “agents” architecture so you can plug in new data sources easily

Outputs comparisons in human-readable format (e.g. Markdown tables)

Easily extensible and customizable

🛠️ How It Works (High Level)

Agent modules — each agent is responsible for fetching specs from a particular website (e.g. GSMArena, official sites, etc.).

Coordinator / orchestrator — sends queries to agents, aggregates their results.

Comparison engine — aligns spec fields across different phones, handles missing values, formats output (table, JSON, etc.).

User interface / script — you run a command (e.g. python main.py --phones iPhone_6s OnePlus_10T) and it prints the comparison.
