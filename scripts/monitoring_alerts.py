import os
import json
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Environment Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Alert thresholds
TICKET_SPIKE_THRESHOLD = float(os.getenv("TICKET_SPIKE_THRESHOLD", "50.0"))  # 50% increase
SENTIMENT_DROP_THRESHOLD = float(os.getenv("SENTIMENT_DROP_THRESHOLD", "-0.3"))  # Significant negative shift
HIGH_SEVERITY_THRESHOLD = int(os.getenv("HIGH_SEVERITY_THRESHOLD", "3"))  # 3+ high severity clusters
VOLUME_SPIKE_THRESHOLD = int(os.getenv("VOLUME_SPIKE_THRESHOLD", "20"))  # 20+ tickets in single theme

# Notification settings
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
GITHUB_ISSUE_ALERTS = os.getenv("GITHUB_ISSUE_ALERTS", "true").lower() == "true"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def die(msg: str):
    logger.error(msg)
    sys.exit(1)

if not SUPABASE_URL: die("Missing SUPABASE_URL")
if not SUPABASE_KEY: die("Missing SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)")

# Initialize clients
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@dataclass
class Alert:
    level: str  # 'critical', 'warning', 'info'
    title: str
    message: str
    metric_value: float
    threshold: float
    recommendation: str
    timestamp: str

class MonitoringSystem:
    def __init__(self):
        self.alerts = []

    def check_ticket_volume_spike(self) -> Optional[Alert]:
        """Monitor for unusual spikes in ticket volume."""
        try:
            # Compare last 24 hours vs previous 24 hours
            now = datetime.now()
            last_24h = (now - timedelta(hours=24)).isoformat()
            previous_24h = (now - timedelta(hours=48)).isoformat()

            # Current period tickets
            current_query = (
                sb.table("raw_gorgias")
                .select("id", count="exact")
                .gte("created_datetime", last_24h)
            )
            current_result = current_query.execute()
            current_count = current_result.count or 0

            # Previous period tickets
            previous_query = (
                sb.table("raw_gorgias")
                .select("id", count="exact")
                .gte("created_datetime", previous_24h)
                .lt("created_datetime", last_24h)
            )
            previous_result = previous_query.execute()
            previous_count = previous_result.count or 0

            if previous_count > 0:
                change_percent = ((current_count - previous_count) / previous_count) * 100

                if change_percent >= TICKET_SPIKE_THRESHOLD:
                    return Alert(
                        level="critical" if change_percent >= 100 else "warning",
                        title=f"Ticket Volume Spike Detected",
                        message=f"Ticket volume increased by {change_percent:.1f}% in last 24h ({current_count} vs {previous_count})",
                        metric_value=change_percent,
                        threshold=TICKET_SPIKE_THRESHOLD,
                        recommendation="Investigate root cause: new product issues, system outages, or marketing campaigns",
                        timestamp=datetime.now().isoformat()
                    )

            return None

        except Exception as e:
            logger.error(f"Error checking ticket volume: {e}")
            return None

    def check_sentiment_deterioration(self) -> Optional[Alert]:
        """Monitor for declining customer sentiment."""
        try:
            # Compare last 7 days vs previous 7 days
            now = datetime.now()
            last_week = (now - timedelta(days=7)).isoformat()
            previous_week = (now - timedelta(days=14)).isoformat()

            # Get current period sentiment
            current_query = (
                sb.table("raw_gorgias")
                .select("ai_sentiment")
                .gte("created_datetime", last_week)
                .not_.is_("ai_sentiment", "null")
            )
            current_result = current_query.execute()
            current_sentiments = [t["ai_sentiment"] for t in (current_result.data or [])]

            # Get previous period sentiment
            previous_query = (
                sb.table("raw_gorgias")
                .select("ai_sentiment")
                .gte("created_datetime", previous_week)
                .lt("created_datetime", last_week)
                .not_.is_("ai_sentiment", "null")
            )
            previous_result = previous_query.execute()
            previous_sentiments = [t["ai_sentiment"] for t in (previous_result.data or [])]

            if len(current_sentiments) >= 10 and len(previous_sentiments) >= 10:
                # Calculate sentiment scores
                def sentiment_score(sentiments):
                    scores = []
                    for s in sentiments:
                        if s == 'positive':
                            scores.append(1.0)
                        elif s == 'negative':
                            scores.append(-1.0)
                        else:
                            scores.append(0.0)
                    return sum(scores) / len(scores) if scores else 0.0

                current_score = sentiment_score(current_sentiments)
                previous_score = sentiment_score(previous_sentiments)
                change = current_score - previous_score

                if change <= SENTIMENT_DROP_THRESHOLD:
                    return Alert(
                        level="critical" if change <= -0.5 else "warning",
                        title="Customer Sentiment Deterioration",
                        message=f"Average sentiment dropped by {abs(change):.2f} points (from {previous_score:.2f} to {current_score:.2f})",
                        metric_value=change,
                        threshold=SENTIMENT_DROP_THRESHOLD,
                        recommendation="Review recent negative feedback themes and implement immediate improvements",
                        timestamp=datetime.now().isoformat()
                    )

            return None

        except Exception as e:
            logger.error(f"Error checking sentiment: {e}")
            return None

    def check_high_severity_clusters(self) -> Optional[Alert]:
        """Monitor for accumulation of high-severity issues."""
        try:
            # Get recent high-severity clusters
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()

            query = (
                sb.table("ticket_clusters")
                .select("*")
                .gte("created_at", cutoff)
                .gte("severity", 4)
                .order("created_at", desc=True)
            )

            result = query.execute()
            high_severity_clusters = result.data or []

            # Get most recent analysis
            if high_severity_clusters:
                latest_time = high_severity_clusters[0]["created_at"]
                recent_clusters = [c for c in high_severity_clusters if c["created_at"] == latest_time]

                if len(recent_clusters) >= HIGH_SEVERITY_THRESHOLD:
                    total_tickets = sum(c["ticket_count"] for c in recent_clusters)
                    themes = [c["theme_name"] for c in recent_clusters[:3]]

                    return Alert(
                        level="critical",
                        title=f"Multiple High-Severity Issues Detected",
                        message=f"{len(recent_clusters)} critical issue categories affecting {total_tickets} tickets: {', '.join(themes)}",
                        metric_value=len(recent_clusters),
                        threshold=HIGH_SEVERITY_THRESHOLD,
                        recommendation="Escalate to product/engineering teams and create immediate action plan",
                        timestamp=datetime.now().isoformat()
                    )

            return None

        except Exception as e:
            logger.error(f"Error checking high severity clusters: {e}")
            return None

    def check_cluster_volume_spikes(self) -> List[Alert]:
        """Monitor for specific themes experiencing volume spikes."""
        alerts = []
        try:
            cutoff = (datetime.now() - timedelta(days=3)).isoformat()

            query = (
                sb.table("ticket_clusters")
                .select("*")
                .gte("created_at", cutoff)
                .gte("ticket_count", VOLUME_SPIKE_THRESHOLD)
                .order("ticket_count", desc=True)
            )

            result = query.execute()
            large_clusters = result.data or []

            # Get most recent analysis
            if large_clusters:
                latest_time = large_clusters[0]["created_at"]
                recent_large = [c for c in large_clusters if c["created_at"] == latest_time]

                for cluster in recent_large[:3]:  # Top 3 largest
                    alerts.append(Alert(
                        level="warning",
                        title=f"High Volume Theme: {cluster['theme_name']}",
                        message=f"'{cluster['theme_name']}' cluster has {cluster['ticket_count']} tickets (severity {cluster['severity']}/5)",
                        metric_value=cluster['ticket_count'],
                        threshold=VOLUME_SPIKE_THRESHOLD,
                        recommendation=f"Review: {cluster['action_items'][:100]}..." if cluster.get('action_items') else "Investigate root cause and implement preventive measures",
                        timestamp=datetime.now().isoformat()
                    ))

        except Exception as e:
            logger.error(f"Error checking cluster volumes: {e}")

        return alerts

    def generate_alert_summary(self) -> str:
        """Generate AI summary of current alerts."""
        if not self.alerts or not openai_client:
            return "No significant alerts detected."

        try:
            alert_text = "\n".join([
                f"- {alert.level.upper()}: {alert.title} - {alert.message}"
                for alert in self.alerts
            ])

            prompt = f"""Analyze these customer support alerts and provide a brief executive summary:

{alert_text}

Provide:
1. Overall severity assessment
2. Top 2-3 priorities for immediate action
3. Potential business impact

Keep response under 150 words and business-focused."""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating alert summary: {e}")
            return f"Alert analysis failed: {str(e)}"

    def format_alerts_report(self) -> str:
        """Format alerts into readable report."""
        if not self.alerts:
            return """
# 🟢 Support Monitoring Status: All Clear

No significant alerts detected in customer support metrics.

**Monitored Metrics:**
- Ticket volume trends
- Customer sentiment changes
- High-severity issue accumulation
- Theme-specific volume spikes

*Next check: Automated monitoring runs continuously*
"""

        critical_alerts = [a for a in self.alerts if a.level == "critical"]
        warning_alerts = [a for a in self.alerts if a.level == "warning"]

        status_emoji = "🔴" if critical_alerts else "🟡"
        status_text = "Critical Issues Detected" if critical_alerts else "Warnings Detected"

        report = f"""
# {status_emoji} Support Monitoring Status: {status_text}

**Alert Summary:** {self.generate_alert_summary()}

## Alert Details

"""

        for alert in sorted(self.alerts, key=lambda x: x.level == "critical", reverse=True):
            emoji = "🔴" if alert.level == "critical" else "🟡" if alert.level == "warning" else "🔵"
            report += f"""
### {emoji} {alert.title}

**Level:** {alert.level.upper()}
**Metric:** {alert.metric_value} (threshold: {alert.threshold})
**Details:** {alert.message}
**Recommendation:** {alert.recommendation}
**Detected:** {alert.timestamp}

---
"""

        report += f"""
## Action Items

### Immediate (Next 2 hours)
"""
        for alert in critical_alerts:
            report += f"- [ ] {alert.recommendation}\n"

        if warning_alerts:
            report += f"""
### Medium Priority (Next 24 hours)
"""
            for alert in warning_alerts[:3]:
                report += f"- [ ] {alert.recommendation}\n"

        report += f"""
## Context
- **Total Alerts:** {len(self.alerts)}
- **Critical:** {len(critical_alerts)}
- **Warnings:** {len(warning_alerts)}
- **Monitoring Period:** Last 24-48 hours

*Automated monitoring by Koio Support Intelligence*
"""

        return report

    def run_monitoring_checks(self) -> str:
        """Execute all monitoring checks and generate report."""
        logger.info("=== Starting Support Monitoring Checks ===")

        # Run all checks
        checks = [
            ("Ticket Volume Spike", self.check_ticket_volume_spike()),
            ("Sentiment Deterioration", self.check_sentiment_deterioration()),
            ("High Severity Clusters", self.check_high_severity_clusters()),
            ("Volume Spikes", self.check_cluster_volume_spikes())
        ]

        for check_name, result in checks:
            logger.info(f"Running {check_name} check...")
            if isinstance(result, list):
                self.alerts.extend(result)
                logger.info(f"  Found {len(result)} alerts")
            elif result:
                self.alerts.append(result)
                logger.info(f"  Found 1 alert")
            else:
                logger.info(f"  No alerts")

        # Generate report
        report = self.format_alerts_report()

        logger.info(f"=== Monitoring Complete: {len(self.alerts)} total alerts ===")

        return report

def main():
    logger.info("=== Koio Support Monitoring Starting ===")

    monitor = MonitoringSystem()
    report = monitor.run_monitoring_checks()

    # Print report
    print("\n" + "="*60)
    print("SUPPORT MONITORING REPORT")
    print("="*60)
    print(report)
    print("="*60)

    # Exit with error code if critical alerts found
    critical_count = len([a for a in monitor.alerts if a.level == "critical"])
    if critical_count > 0:
        logger.warning(f"Exiting with error code due to {critical_count} critical alerts")
        sys.exit(1)

    return True

if __name__ == "__main__":
    main()