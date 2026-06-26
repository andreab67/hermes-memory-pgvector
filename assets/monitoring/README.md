# Hexus MCP Monitoring Dashboard

This directory contains a pre-configured Grafana dashboard template for monitoring a running Hexus MCP instance.

## Contents
* `hexus_dashboard.json`: The Grafana dashboard configuration (31 panels across 8 sections).

## Setup Instructions

### 1. Prometheus Scrape Configuration
To fetch metrics from your Hexus MCP server, add a scrape job to your `prometheus.yml` configuration:

```yaml
scrape_configs:
  - job_name: "hexus"
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ["<hexus-mcp-host>:8000"]
```

Replace `<hexus-mcp-host>` with the IP or hostname of your running Hexus MCP container or service.

### 2. Import into Grafana
1. In Grafana, click the **+** icon in the top right and select **Import dashboard**.
2. Upload the `hexus_dashboard.json` file or copy-paste its contents.
3. Select your Prometheus datasource when prompted.
4. Click **Import**.

## Dashboard Sections
* **Service Health**: Database status (liveness), total memory entries, total conversation turns, and turns/memory ratio.
* **Growth Over Time**: Cumulative metrics showing database size expansion.
* **Activity Rates**: Per-second ingestion rates for memory and conversation turns.
* **Memory Feedback & Recalls**: Negative/positive reinforcement indicators (confirms, rejects) and recall event rates.
* **Agent Profiling**: Visual bar gauges showing metrics broken down per `agent_identity`.
* **Delegations & Semantic Entities**: Unique entity counts and mentions extracted from memories and conversations.
* **DB Health & Uptime**: Availability timelines and dual-axis correlation charts.
* **Scrape Health**: Prometheus performance and scraper latencies.
