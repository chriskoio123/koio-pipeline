import os
import json
import sys
import logging
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# --- Environment Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

# Email Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_USER)

# Report Configuration
REPORT_DAYS = int(os.getenv("REPORT_ANALYSIS_DAYS", "30"))

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
if not EMAIL_TO: die("Missing EMAIL_TO - recipient email address required")
if not EMAIL_USER: die("Missing EMAIL_USER - sender email address required")
if not EMAIL_PASSWORD: die("Missing EMAIL_PASSWORD - email authentication required")

# Initialize Supabase client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

class EmailReporter:
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.email_user = EMAIL_USER
        self.email_password = EMAIL_PASSWORD
        self.email_to = EMAIL_TO
        self.email_from = EMAIL_FROM

    def test_email_connection(self) -> bool:
        """Test email server connection."""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.quit()
            logger.info("✅ Email server connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Email server connection failed: {e}")
            return False

    def prepare_metrics_data(self) -> List[Dict[str, Any]]:
        """Fetch and prepare metrics data for export."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=REPORT_DAYS)).isoformat()

            logger.info(f"Fetching metrics data from last {REPORT_DAYS} days...")

            # Get ticket data
            tickets_query = (
                sb.table("raw_gorgias")
                .select("id, created_datetime, ai_sentiment, ai_priority, ai_labels")
                .gte("created_datetime", cutoff_date)
                .not_.is_("ai_sentiment", "null")
                .order("created_datetime", desc=True)
            )

            tickets_result = tickets_query.execute()
            tickets = tickets_result.data or []

            # Get cluster data
            clusters_query = (
                sb.table("ticket_clusters")
                .select("*")
                .gte("created_at", cutoff_date)
                .order("created_at", desc=True)
                .limit(10)
            )

            clusters_result = clusters_query.execute()
            clusters = clusters_result.data or []

            # Prepare summary metrics
            total_tickets = len(tickets)

            if tickets:
                sentiment_dist = {}
                for ticket in tickets:
                    sentiment = ticket.get('ai_sentiment', 'neutral')
                    sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1

                avg_sentiment = self._calculate_avg_sentiment([t.get('ai_sentiment', 'neutral') for t in tickets])
            else:
                sentiment_dist = {'positive': 0, 'neutral': 0, 'negative': 0}
                avg_sentiment = 0.0

            # Prepare data for Sheets
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            summary_data = [
                {
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Period_Days": REPORT_DAYS,
                    "Total_Tickets": total_tickets,
                    "Avg_Sentiment": round(avg_sentiment, 2),
                    "Positive_Tickets": sentiment_dist.get('positive', 0),
                    "Neutral_Tickets": sentiment_dist.get('neutral', 0),
                    "Negative_Tickets": sentiment_dist.get('negative', 0),
                    "Top_Clusters": len(clusters),
                    "High_Severity_Count": len([c for c in clusters if c.get('severity', 3) >= 4])
                }
            ]

            cluster_data = []
            for i, cluster in enumerate(clusters[:5], 1):
                cluster_data.append({
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Rank": i,
                    "Theme_Name": cluster.get('theme_name', ''),
                    "Ticket_Count": cluster.get('ticket_count', 0),
                    "Severity": cluster.get('severity', 3),
                    "Sentiment_Trend": cluster.get('sentiment_trend', 'neutral'),
                    "Avg_Sentiment": round(cluster.get('avg_sentiment', 0), 2),
                    "Summary": (cluster.get('summary', '')[:100] + '...' if len(cluster.get('summary', '')) > 100 else cluster.get('summary', ''))
                })

            return summary_data, cluster_data

        except Exception as e:
            logger.error(f"Error preparing metrics data: {e}")
            return [], []

    def _calculate_avg_sentiment(self, sentiments: List[str]) -> float:
        """Calculate average sentiment score."""
        if not sentiments:
            return 0.0

        scores = []
        for sentiment in sentiments:
            if sentiment == 'positive':
                scores.append(1.0)
            elif sentiment == 'negative':
                scores.append(-1.0)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores)

    def create_email_html_format(self, summary_data: List[Dict], cluster_data: List[Dict]) -> str:
        """Create HTML formatted email report."""
        try:
            if not summary_data:
                return "<p>No data available for report generation.</p>"

            data = summary_data[0]

            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                    .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
                    .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                    .alert {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }}
                    .positive {{ color: #27ae60; font-weight: bold; }}
                    .negative {{ color: #e74c3c; font-weight: bold; }}
                    .neutral {{ color: #95a5a6; font-weight: bold; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #34495e; color: white; }}
                    .trend-up {{ color: #27ae60; }}
                    .trend-down {{ color: #e74c3c; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🏢 KOIO CUSTOMER SUPPORT INTELLIGENCE</h1>
                    <p>Weekly Business Intelligence Report</p>
                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
                </div>

                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="metrics">
                        <p><strong>Analysis Period:</strong> Last {data['Period_Days']} days</p>
                        <p><strong>Total Support Tickets:</strong> {data['Total_Tickets']}</p>
                        <p><strong>Customer Sentiment Score:</strong> {data['Avg_Sentiment']} (-1 to +1 scale)</p>
                    </div>
                </div>"""

            # Sentiment Analysis
            total = data['Total_Tickets']
            if total > 0:
                pos_pct = round((data['Positive_Tickets'] / total) * 100, 1)
                neg_pct = round((data['Negative_Tickets'] / total) * 100, 1)
                neu_pct = round((data['Neutral_Tickets'] / total) * 100, 1)

                html += f"""
                <div class="section">
                    <h2>😊 Customer Sentiment Breakdown</h2>
                    <p><span class="positive">{pos_pct}% Positive</span> ({data['Positive_Tickets']} tickets)</p>
                    <p><span class="neutral">{neu_pct}% Neutral</span> ({data['Neutral_Tickets']} tickets)</p>
                    <p><span class="negative">{neg_pct}% Negative</span> ({data['Negative_Tickets']} tickets)</p>

                    {"<div class='alert'>📈 <strong>Positive Trend:</strong> Customer satisfaction is above average!</div>" if data['Avg_Sentiment'] > 0.2 else
                     "<div class='alert'>📉 <strong>Attention Needed:</strong> Customer sentiment is declining.</div>" if data['Avg_Sentiment'] < -0.2 else
                     "<div class='alert'>📊 <strong>Stable:</strong> Customer sentiment is neutral/stable.</div>"}
                </div>"""

            # Top Support Themes
            if cluster_data:
                html += """
                <div class="section">
                    <h2>🎯 Top Support Themes</h2>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Theme</th>
                            <th>Tickets</th>
                            <th>Severity</th>
                            <th>Trend</th>
                            <th>Summary</th>
                        </tr>"""

                for cluster in cluster_data[:5]:
                    severity_color = "#e74c3c" if cluster['Severity'] >= 4 else "#f39c12" if cluster['Severity'] >= 3 else "#27ae60"
                    html += f"""
                        <tr>
                            <td><strong>{cluster['Rank']}</strong></td>
                            <td>{cluster['Theme_Name']}</td>
                            <td>{cluster['Ticket_Count']}</td>
                            <td style="color: {severity_color};">{cluster['Severity']}/5</td>
                            <td>{cluster['Sentiment_Trend']}</td>
                            <td>{cluster['Summary'][:100]}{'...' if len(cluster['Summary']) > 100 else ''}</td>
                        </tr>"""

                html += "</table></div>"

            # Action Items
            html += """
                <div class="section">
                    <h2>🎯 Recommended Actions</h2>"""

            if cluster_data:
                high_severity = [c for c in cluster_data if c['Severity'] >= 4]
                high_volume = [c for c in cluster_data if c['Ticket_Count'] >= 10]

                if high_severity:
                    html += f"<p><strong>🚨 Priority 1:</strong> Address {len(high_severity)} high-severity issues</p><ul>"
                    for cluster in high_severity[:2]:
                        html += f"<li>{cluster['Theme_Name']} - {cluster['Ticket_Count']} tickets (severity {cluster['Severity']})</li>"
                    html += "</ul>"

                if high_volume:
                    html += f"<p><strong>📊 Priority 2:</strong> Optimize {len(high_volume)} high-volume themes</p><ul>"
                    for cluster in high_volume[:2]:
                        html += f"<li>{cluster['Theme_Name']} - {cluster['Ticket_Count']} tickets</li>"
                    html += "</ul>"

                html += "<p><strong>📈 Priority 3:</strong> Continue monitoring sentiment trends and customer feedback</p>"
            else:
                html += "<p>📊 Continue monitoring support metrics and customer satisfaction</p>"

            html += """
                </div>

                <div class="section">
                    <h2>🔗 Quick Links</h2>
                    <p>📊 <a href="https://app.supabase.com">Live Dashboard</a></p>
                    <p>🔄 <a href="https://github.com/chriskoio123/koio-pipeline/actions">Automation Status</a></p>
                    <p>📅 Next Report: {next_report}</p>
                </div>

                <div style="margin-top: 30px; padding: 15px; background-color: #ecf0f1; text-align: center; font-size: 12px; color: #7f8c8d;">
                    <p>🤖 This report was automatically generated by the KOIO Support Intelligence Pipeline</p>
                    <p>For questions or support, contact your development team</p>
                </div>
            </body>
            </html>""".format(next_report=(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'))

            return html

        except Exception as e:
            logger.error(f"Error creating email HTML format: {e}")
            return f"<p>Error formatting report: {str(e)}</p>"

            if summary_data:
                data = summary_data[0]

                # Key Metrics Section - formatted for easy reading
                lines.append("=== KEY METRICS ===")
                lines.append(f"Report Period\t{data['Period_Days']} days")
                lines.append(f"Total Tickets\t{data['Total_Tickets']}")
                lines.append(f"Average Sentiment\t{data['Avg_Sentiment']}")
                lines.append(f"Positive Tickets\t{data['Positive_Tickets']}")
                lines.append(f"Neutral Tickets\t{data['Neutral_Tickets']}")
                lines.append(f"Negative Tickets\t{data['Negative_Tickets']}")
                lines.append(f"High Severity Issues\t{data['High_Severity_Count']}")
                lines.append("")

                # Quick Summary
                total = data['Total_Tickets']
                if total > 0:
                    pos_pct = round((data['Positive_Tickets'] / total) * 100, 1)
                    neg_pct = round((data['Negative_Tickets'] / total) * 100, 1)
                    lines.append("=== QUICK INSIGHTS ===")
                    lines.append(f"Customer Satisfaction\t{pos_pct}% positive, {neg_pct}% negative")

                    if data['Avg_Sentiment'] > 0.2:
                        lines.append("Overall Sentiment\tPositive trend 📈")
                    elif data['Avg_Sentiment'] < -0.2:
                        lines.append("Overall Sentiment\tNegative trend 📉")
                    else:
                        lines.append("Overall Sentiment\tNeutral/stable")

                    lines.append("")

            # Top Issues Section
            if cluster_data:
                lines.append("=== TOP SUPPORT THEMES ===")
                lines.append("Rank\tTheme\tTickets\tSeverity\tSentiment\tSummary")

                for cluster in cluster_data[:5]:
                    summary_short = cluster['Summary'][:50] + "..." if len(cluster['Summary']) > 50 else cluster['Summary']
                    lines.append(f"{cluster['Rank']}\t{cluster['Theme_Name']}\t{cluster['Ticket_Count']}\t{cluster['Severity']}/5\t{cluster['Sentiment_Trend']}\t{summary_short}")

                lines.append("")

            # Action Items
            lines.append("=== RECOMMENDED ACTIONS ===")
            if cluster_data:
                high_severity = [c for c in cluster_data if c['Severity'] >= 4]
                high_volume = [c for c in cluster_data if c['Ticket_Count'] >= 10]

                if high_severity:
                    lines.append(f"🚨 Priority 1\tAddress {len(high_severity)} high-severity issues")
                    for cluster in high_severity[:2]:
                        lines.append(f"  • {cluster['Theme_Name']}\t{cluster['Ticket_Count']} tickets (severity {cluster['Severity']})")

                if high_volume:
                    lines.append(f"📊 Priority 2\tOptimize {len(high_volume)} high-volume themes")
                    for cluster in high_volume[:2]:
                        lines.append(f"  • {cluster['Theme_Name']}\t{cluster['Ticket_Count']} tickets")

                lines.append("📈 Priority 3\tMonitor sentiment trends and customer feedback")
            else:
                lines.append("📊 Monitor\tContinue tracking support metrics")

            lines.append("")
            lines.append("=== DATA ACCESS ===")
            lines.append("Live Dashboard\thttps://app.supabase.com")
            lines.append("GitHub Actions\thttps://github.com/chriskoio123/koio-pipeline/actions")
            lines.append(f"Next Report\t{(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error creating copy-paste format: {e}")
            return f"Error formatting data: {str(e)}"

    def create_sheets_format_data(self, summary_data: List[Dict], cluster_data: List[Dict]) -> str:
        """Create data in a format suitable for Google Sheets."""
        try:
            output = []

            # Add title
            output.append("=== KOIO CUSTOMER SUPPORT INTELLIGENCE REPORT ===")
            output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            output.append("")

            # Summary Section
            output.append("SUMMARY METRICS")
            output.append("-" * 50)
            if summary_data:
                data = summary_data[0]
                output.append(f"Report Period: Last {data['Period_Days']} days")
                output.append(f"Total Tickets: {data['Total_Tickets']}")
                output.append(f"Average Sentiment: {data['Avg_Sentiment']} (-1 to 1)")
                output.append(f"Positive: {data['Positive_Tickets']}, Neutral: {data['Neutral_Tickets']}, Negative: {data['Negative_Tickets']}")
                output.append(f"Active Clusters: {data['Top_Clusters']}")
                output.append(f"High Severity Issues: {data['High_Severity_Count']}")
            output.append("")

            # Cluster Section
            output.append("TOP SUPPORT THEMES")
            output.append("-" * 50)
            for cluster in cluster_data:
                output.append(f"{cluster['Rank']}. {cluster['Theme_Name']}")
                output.append(f"   Tickets: {cluster['Ticket_Count']} | Severity: {cluster['Severity']}/5 | Sentiment: {cluster['Sentiment_Trend']}")
                output.append(f"   Summary: {cluster['Summary']}")
                output.append("")

            # Instructions
            output.append("DATA ACCESS")
            output.append("-" * 50)
            output.append("• View live data: https://app.supabase.com")
            output.append("• Download CSV reports: GitHub Actions artifacts")
            output.append("• API access: Contact admin for credentials")
            output.append("")
            output.append(f"Next update: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Error creating sheets format: {e}")
            return f"Error formatting data: {str(e)}"

    def send_email_report(self, html_content: str, subject: str) -> bool:
        """Send HTML email report."""
        try:
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_from
            msg['To'] = self.email_to

            # Create HTML part
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_from, [self.email_to], text)
            server.quit()

            logger.info(f"✅ Email report sent successfully to {self.email_to}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False

    def generate_and_send_report(self) -> str:
        """Generate and send HTML email report."""
        try:
            logger.info("📧 Generating email business intelligence report...")

            # Test email connection first
            if not self.test_email_connection():
                return "❌ Email connection failed - check email configuration"

            summary_data, cluster_data = self.prepare_metrics_data()

            if not summary_data:
                return "❌ No data available for report generation"

            # Generate HTML content
            html_content = self.create_email_html_format(summary_data, cluster_data)

            # Create email subject
            report_date = datetime.now().strftime('%Y-%m-%d')
            total_tickets = summary_data[0]['Total_Tickets'] if summary_data else 0
            subject = f"📊 KOIO Support Intelligence Report - {report_date} ({total_tickets} tickets)"

            # Send email
            if self.send_email_report(html_content, subject):
                # Save backup files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._create_csv_files(summary_data, cluster_data, timestamp)

                return f"""
✅ EMAIL REPORT SENT SUCCESSFULLY!

📧 EMAIL DETAILS:
- To: {self.email_to}
- Subject: {subject}
- Content: Professional HTML business intelligence report
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

📁 BACKUP FILES CREATED:
- summary_{timestamp}.csv (summary data)
- clusters_{timestamp}.csv (cluster analysis)

📊 REPORT SUMMARY:
- Analysis Period: {summary_data[0]['Period_Days']} days
- Total Tickets: {summary_data[0]['Total_Tickets']}
- Sentiment Score: {summary_data[0]['Avg_Sentiment']}
- Active Clusters: {len(cluster_data)}

💡 Check your email inbox for the detailed HTML report!
"""
            else:
                return "❌ Failed to send email report - check logs for details"

        except Exception as e:
            logger.error(f"Error generating email report: {e}")
            return f"❌ Email report generation failed: {str(e)}"

    def _create_csv_files(self, summary_data: List[Dict], cluster_data: List[Dict], timestamp: str):
        """Create CSV files for easy Google Sheets import."""
        try:
            import csv

            # Summary CSV
            if summary_data:
                with open(f"summary_{timestamp}.csv", 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                    writer.writeheader()
                    writer.writerows(summary_data)

            # Clusters CSV
            if cluster_data:
                with open(f"clusters_{timestamp}.csv", 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=cluster_data[0].keys())
                    writer.writeheader()
                    writer.writerows(cluster_data)

            logger.info(f"CSV files created: summary_{timestamp}.csv, clusters_{timestamp}.csv")

        except Exception as e:
            logger.error(f"Error creating CSV files: {e}")

def main():
    logger.info("=== KOIO Email Business Intelligence Report Starting ===")

    reporter = EmailReporter()
    result = reporter.generate_and_send_report()

    print("\n" + "="*60)
    print("EMAIL BUSINESS INTELLIGENCE REPORT")
    print("="*60)
    print(result)
    print("="*60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)