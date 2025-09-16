import os
import time
import json
import sys
import traceback
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Environment Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Report Configuration
REPORT_DAYS = int(os.getenv("REPORT_ANALYSIS_DAYS", "30"))
COMPARISON_DAYS = int(os.getenv("REPORT_COMPARISON_DAYS", "60"))
MIN_TICKETS_FOR_TREND = int(os.getenv("MIN_TICKETS_FOR_TREND", "5"))

# AI Configuration
GPT_MODEL = os.getenv("REPORT_GPT_MODEL", "gpt-4o-mini")
MAX_REPORT_TOKENS = int(os.getenv("MAX_REPORT_TOKENS", "1000"))

# Google Sheets Configuration (optional)
GOOGLE_SHEETS_URL = os.getenv("GOOGLE_SHEETS_URL", "")
SHEETS_EXPORT_ENABLED = os.getenv("SHEETS_EXPORT_ENABLED", "false").lower() == "true"

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
if not OPENAI_API_KEY: die("Missing OPENAI_API_KEY")

# Initialize clients
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@dataclass
class BusinessMetrics:
    total_tickets: int
    avg_sentiment: float
    sentiment_distribution: Dict[str, int]
    top_clusters: List[Dict[str, Any]]
    high_severity_clusters: List[Dict[str, Any]]
    ticket_volume_trend: str
    sentiment_trend: str
    emerging_issues: List[str]
    resolved_themes: List[str]

@dataclass
class TrendAnalysis:
    period_start: str
    period_end: str
    ticket_count_change: float
    sentiment_change: float
    new_clusters: List[str]
    growing_clusters: List[str]
    declining_clusters: List[str]
    anomalies: List[str]

class BusinessReportGenerator:
    def __init__(self):
        self.current_metrics = None
        self.previous_metrics = None
        self.trend_analysis = None

    def fetch_ticket_metrics(self, days_back: int) -> BusinessMetrics:
        """Fetch comprehensive ticket metrics for analysis."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()

            logger.info(f"Fetching ticket metrics from last {days_back} days...")

            # Get ticket data with AI enrichment
            tickets_query = (
                sb.table("raw_gorgias")
                .select("id, created_datetime, ai_sentiment, ai_priority, ai_labels")
                .gte("created_datetime", cutoff_date)
                .not_.is_("ai_sentiment", "null")
                .order("created_datetime", desc=True)
            )

            tickets_result = tickets_query.execute()
            tickets = tickets_result.data or []

            if not tickets:
                logger.warning(f"No tickets found for last {days_back} days")
                return BusinessMetrics(0, 0.0, {}, [], [], "stable", "stable", [], [])

            # Calculate sentiment metrics
            sentiments = [t.get('ai_sentiment', 'neutral') for t in tickets]
            sentiment_dist = {}
            sentiment_scores = []

            for sentiment in sentiments:
                sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1
                if sentiment == 'positive':
                    sentiment_scores.append(1.0)
                elif sentiment == 'negative':
                    sentiment_scores.append(-1.0)
                else:
                    sentiment_scores.append(0.0)

            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

            # Get cluster data
            clusters_query = (
                sb.table("ticket_clusters")
                .select("*")
                .gte("created_at", cutoff_date)
                .order("created_at", desc=True)
                .order("ticket_count", desc=True)
            )

            clusters_result = clusters_query.execute()
            all_clusters = clusters_result.data or []

            # Get latest clusters (most recent analysis)
            if all_clusters:
                latest_analysis_time = all_clusters[0]['created_at']
                latest_clusters = [c for c in all_clusters if c['created_at'] == latest_analysis_time]
            else:
                latest_clusters = []

            # Identify top and high-severity clusters
            top_clusters = sorted(latest_clusters, key=lambda x: x['ticket_count'], reverse=True)[:5]
            high_severity_clusters = [c for c in latest_clusters if c.get('severity', 3) >= 4]

            logger.info(f"Found {len(tickets)} tickets and {len(latest_clusters)} clusters")

            return BusinessMetrics(
                total_tickets=len(tickets),
                avg_sentiment=avg_sentiment,
                sentiment_distribution=sentiment_dist,
                top_clusters=top_clusters,
                high_severity_clusters=high_severity_clusters,
                ticket_volume_trend="stable",  # Will be calculated in trend analysis
                sentiment_trend="stable",      # Will be calculated in trend analysis
                emerging_issues=[],            # Will be identified in trend analysis
                resolved_themes=[]             # Will be identified in trend analysis
            )

        except Exception as e:
            logger.error(f"Error fetching ticket metrics: {e}")
            traceback.print_exc()
            return BusinessMetrics(0, 0.0, {}, [], [], "stable", "stable", [], [])

    def perform_trend_analysis(self) -> TrendAnalysis:
        """Compare current period with previous period for trends."""
        try:
            logger.info("Performing trend analysis...")

            if not self.current_metrics or not self.previous_metrics:
                logger.warning("Insufficient data for trend analysis")
                return TrendAnalysis("", "", 0.0, 0.0, [], [], [], [])

            # Calculate changes
            ticket_change = 0.0
            if self.previous_metrics.total_tickets > 0:
                ticket_change = ((self.current_metrics.total_tickets - self.previous_metrics.total_tickets)
                               / self.previous_metrics.total_tickets) * 100

            sentiment_change = self.current_metrics.avg_sentiment - self.previous_metrics.avg_sentiment

            # Analyze cluster trends
            current_themes = {c['theme_name']: c['ticket_count'] for c in self.current_metrics.top_clusters}
            previous_themes = {c['theme_name']: c['ticket_count'] for c in self.previous_metrics.top_clusters}

            new_clusters = [theme for theme in current_themes if theme not in previous_themes]

            growing_clusters = []
            declining_clusters = []

            for theme, current_count in current_themes.items():
                if theme in previous_themes:
                    previous_count = previous_themes[theme]
                    if current_count > previous_count * 1.5:  # 50% growth threshold
                        growing_clusters.append(theme)
                    elif current_count < previous_count * 0.5:  # 50% decline threshold
                        declining_clusters.append(theme)

            # Identify anomalies
            anomalies = []
            if abs(ticket_change) > 50:  # Major volume change
                direction = "increase" if ticket_change > 0 else "decrease"
                anomalies.append(f"Major ticket volume {direction}: {ticket_change:.1f}%")

            if abs(sentiment_change) > 0.5:  # Major sentiment shift
                direction = "improvement" if sentiment_change > 0 else "decline"
                anomalies.append(f"Significant sentiment {direction}: {sentiment_change:.2f}")

            period_start = (datetime.now() - timedelta(days=REPORT_DAYS)).strftime("%Y-%m-%d")
            period_end = datetime.now().strftime("%Y-%m-%d")

            return TrendAnalysis(
                period_start=period_start,
                period_end=period_end,
                ticket_count_change=ticket_change,
                sentiment_change=sentiment_change,
                new_clusters=new_clusters,
                growing_clusters=growing_clusters,
                declining_clusters=declining_clusters,
                anomalies=anomalies
            )

        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            traceback.print_exc()
            return TrendAnalysis("", "", 0.0, 0.0, [], [], [], [])

    def generate_executive_summary(self) -> str:
        """Generate AI-powered executive summary."""
        try:
            logger.info("Generating executive summary...")

            if not self.current_metrics or not self.trend_analysis:
                return "Insufficient data for executive summary generation."

            # Prepare data for AI analysis
            summary_prompt = f"""Generate a concise executive summary for customer support insights:

CURRENT PERIOD METRICS:
- Total Tickets: {self.current_metrics.total_tickets}
- Average Sentiment: {self.current_metrics.avg_sentiment:.2f} (-1 to 1 scale)
- Sentiment Distribution: {self.current_metrics.sentiment_distribution}

TOP THEMES:
{chr(10).join([f"• {c['theme_name']}: {c['ticket_count']} tickets (severity {c['severity']}/5)" for c in self.current_metrics.top_clusters[:3]])}

HIGH PRIORITY ISSUES:
{chr(10).join([f"• {c['theme_name']}: {c['ticket_count']} tickets" for c in self.current_metrics.high_severity_clusters[:3]])}

TRENDS:
- Ticket Volume Change: {self.trend_analysis.ticket_count_change:.1f}%
- Sentiment Change: {self.trend_analysis.sentiment_change:.2f}
- New Issues: {', '.join(self.trend_analysis.new_clusters) if self.trend_analysis.new_clusters else 'None'}
- Growing Concerns: {', '.join(self.trend_analysis.growing_clusters) if self.trend_analysis.growing_clusters else 'None'}

ANOMALIES:
{chr(10).join([f"• {a}" for a in self.trend_analysis.anomalies]) if self.trend_analysis.anomalies else "• No significant anomalies detected"}

Provide a business-focused summary in 3-4 bullet points covering:
1. Overall support health
2. Key themes requiring attention
3. Notable trends or changes
4. Recommended actions

Keep it concise and actionable for executives."""

            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=MAX_REPORT_TOKENS,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Executive summary generation failed: {str(e)}"

    def format_detailed_report(self, executive_summary: str) -> str:
        """Format comprehensive business report."""
        try:
            report = f"""
# Customer Support Intelligence Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
Analysis Period: {self.trend_analysis.period_start} to {self.trend_analysis.period_end}

## Executive Summary
{executive_summary}

## Key Metrics

### Volume & Sentiment
- **Total Tickets**: {self.current_metrics.total_tickets}
- **Average Sentiment**: {self.current_metrics.avg_sentiment:.2f} ({self._sentiment_to_label(self.current_metrics.avg_sentiment)})
- **Sentiment Distribution**:
  - Positive: {self.current_metrics.sentiment_distribution.get('positive', 0)}
  - Neutral: {self.current_metrics.sentiment_distribution.get('neutral', 0)}
  - Negative: {self.current_metrics.sentiment_distribution.get('negative', 0)}

### Trend Analysis
- **Ticket Volume Change**: {self.trend_analysis.ticket_count_change:+.1f}%
- **Sentiment Change**: {self.trend_analysis.sentiment_change:+.2f}

## Issue Categories

### Top Support Themes
"""

            for i, cluster in enumerate(self.current_metrics.top_clusters[:5], 1):
                report += f"""
**{i}. {cluster['theme_name']}**
- Tickets: {cluster['ticket_count']}
- Severity: {cluster['severity']}/5
- Sentiment: {cluster['sentiment_trend']}
- Summary: {cluster['summary'][:100]}...
- Action Items: {cluster['action_items'][:100]}...
"""

            if self.current_metrics.high_severity_clusters:
                report += f"""
### High Priority Issues (Severity 4-5)
"""
                for cluster in self.current_metrics.high_severity_clusters:
                    report += f"""
**{cluster['theme_name']}** ({cluster['ticket_count']} tickets)
- {cluster['summary'][:150]}...
"""

            report += f"""
## Trend Insights

### Growth Areas
"""
            if self.trend_analysis.growing_clusters:
                for theme in self.trend_analysis.growing_clusters:
                    report += f"- {theme} (increasing volume)\n"
            else:
                report += "- No significant growth trends detected\n"

            report += f"""
### New Issues
"""
            if self.trend_analysis.new_clusters:
                for theme in self.trend_analysis.new_clusters:
                    report += f"- {theme} (newly emerged)\n"
            else:
                report += "- No new issue categories detected\n"

            if self.trend_analysis.anomalies:
                report += f"""
### Anomalies & Alerts
"""
                for anomaly in self.trend_analysis.anomalies:
                    report += f"⚠️ {anomaly}\n"

            report += f"""
## Recommendations

### Immediate Actions
"""

            # Generate recommendations based on data
            recommendations = []

            if self.current_metrics.high_severity_clusters:
                recommendations.append(f"Address {len(self.current_metrics.high_severity_clusters)} high-severity issue categories")

            if self.trend_analysis.ticket_count_change > 20:
                recommendations.append("Investigate cause of ticket volume spike")
            elif self.trend_analysis.ticket_count_change < -20:
                recommendations.append("Analyze factors contributing to reduced ticket volume")

            if self.current_metrics.avg_sentiment < -0.3:
                recommendations.append("Focus on improving customer satisfaction in negative sentiment areas")

            if self.trend_analysis.growing_clusters:
                recommendations.append(f"Monitor growing issues: {', '.join(self.trend_analysis.growing_clusters[:2])}")

            if not recommendations:
                recommendations.append("Continue monitoring current support trends")

            for i, rec in enumerate(recommendations[:5], 1):
                report += f"{i}. {rec}\n"

            report += f"""
### Process Improvements
1. Review workflow efficiency for top volume categories
2. Update knowledge base based on common themes
3. Consider proactive outreach for high-severity issues
4. Implement prevention strategies for recurring problems

---
*Report generated by Koio AI Customer Support Intelligence*
*Next scheduled update: {(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")}*
"""

            return report

        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"Report formatting failed: {str(e)}"

    def _sentiment_to_label(self, score: float) -> str:
        """Convert sentiment score to human-readable label."""
        if score > 0.3:
            return "Positive"
        elif score < -0.3:
            return "Negative"
        else:
            return "Neutral"

    def export_to_csv(self, report: str) -> str:
        """Export metrics to CSV format for further analysis."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create summary data
            summary_data = {
                'metric': [
                    'Total Tickets',
                    'Average Sentiment',
                    'Positive Tickets',
                    'Neutral Tickets',
                    'Negative Tickets',
                    'High Severity Clusters',
                    'Ticket Volume Change (%)',
                    'Sentiment Change'
                ],
                'value': [
                    self.current_metrics.total_tickets,
                    f"{self.current_metrics.avg_sentiment:.2f}",
                    self.current_metrics.sentiment_distribution.get('positive', 0),
                    self.current_metrics.sentiment_distribution.get('neutral', 0),
                    self.current_metrics.sentiment_distribution.get('negative', 0),
                    len(self.current_metrics.high_severity_clusters),
                    f"{self.trend_analysis.ticket_count_change:.1f}",
                    f"{self.trend_analysis.sentiment_change:.2f}"
                ]
            }

            df_summary = pd.DataFrame(summary_data)
            summary_file = f"support_summary_{timestamp}.csv"
            df_summary.to_csv(summary_file, index=False)

            # Create clusters data
            if self.current_metrics.top_clusters:
                clusters_data = []
                for cluster in self.current_metrics.top_clusters:
                    clusters_data.append({
                        'theme_name': cluster['theme_name'],
                        'ticket_count': cluster['ticket_count'],
                        'severity': cluster['severity'],
                        'sentiment_trend': cluster['sentiment_trend'],
                        'avg_sentiment': cluster.get('avg_sentiment', 0),
                        'summary': cluster['summary'][:200] + '...' if len(cluster['summary']) > 200 else cluster['summary']
                    })

                df_clusters = pd.DataFrame(clusters_data)
                clusters_file = f"support_clusters_{timestamp}.csv"
                df_clusters.to_csv(clusters_file, index=False)

                logger.info(f"Exported data to {summary_file} and {clusters_file}")
                return f"Data exported to {summary_file} and {clusters_file}"
            else:
                logger.info(f"Exported summary to {summary_file}")
                return f"Summary data exported to {summary_file}"

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return f"CSV export failed: {str(e)}"

    def generate_full_report(self) -> str:
        """Main report generation pipeline."""
        try:
            logger.info("=== Starting Business Report Generation ===")

            # Fetch current period metrics
            logger.info("Fetching current period metrics...")
            self.current_metrics = self.fetch_ticket_metrics(REPORT_DAYS)

            # Fetch comparison period metrics
            logger.info("Fetching comparison period metrics...")
            self.previous_metrics = self.fetch_ticket_metrics(COMPARISON_DAYS)

            # Perform trend analysis
            self.trend_analysis = self.perform_trend_analysis()

            # Generate executive summary
            executive_summary = self.generate_executive_summary()

            # Format detailed report
            detailed_report = self.format_detailed_report(executive_summary)

            # Export data
            export_result = self.export_to_csv(detailed_report)

            logger.info("=== Business Report Generation Complete ===")
            logger.info(f"Export result: {export_result}")

            return detailed_report

        except Exception as e:
            logger.error(f"Error in report generation: {e}")
            traceback.print_exc()
            return f"Report generation failed: {str(e)}"

def main():
    logger.info("=== Koio Business Intelligence Report Starting ===")

    generator = BusinessReportGenerator()
    report = generator.generate_full_report()

    # Print report to stdout for GitHub Actions
    print("\n" + "="*60)
    print("CUSTOMER SUPPORT INTELLIGENCE REPORT")
    print("="*60)
    print(report)
    print("="*60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)