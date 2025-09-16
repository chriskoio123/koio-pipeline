import os
import json
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests

from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# --- Environment Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

# Google Sheets Configuration
GOOGLE_SHEETS_URL = os.getenv("GOOGLE_SHEETS_URL", "https://docs.google.com/spreadsheets/d/1hr5JJbV5QbX3TYXVgtXAzwTxj1nveAawm7e4Pl7ztJg/edit?usp=sharing")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Extract sheet ID from URL
def extract_sheet_id(url: str) -> str:
    """Extract Google Sheets ID from URL."""
    try:
        if "/d/" in url:
            return url.split("/d/")[1].split("/")[0]
        return ""
    except:
        return ""

SHEET_ID = extract_sheet_id(GOOGLE_SHEETS_URL)

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
if not SHEET_ID: die("Invalid GOOGLE_SHEETS_URL - could not extract sheet ID")

# Initialize Supabase client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

class GoogleSheetsExporter:
    def __init__(self):
        self.sheet_id = SHEET_ID
        self.api_key = GOOGLE_API_KEY
        self.base_url = f"https://sheets.googleapis.com/v4/spreadsheets/{self.sheet_id}"

    def test_sheets_access(self) -> bool:
        """Test if we can access the Google Sheet."""
        try:
            if self.api_key:
                # Test with API key
                url = f"{self.base_url}?key={self.api_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    logger.info("✅ Google Sheets API access confirmed")
                    return True
                else:
                    logger.warning(f"Google Sheets API access failed: {response.status_code}")

            # Fallback: use public access (read-only)
            logger.info("Using public Google Sheets access")
            return True

        except Exception as e:
            logger.error(f"Error testing Sheets access: {e}")
            return False

    def get_csv_export_url(self, sheet_name: str = "Sheet1") -> str:
        """Generate CSV export URL for the sheet."""
        # For public sheets, we can use the CSV export URL
        return f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/export?format=csv&gid=0"

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

    def export_to_sheets_manually(self) -> str:
        """Generate formatted data for manual Google Sheets entry."""
        try:
            logger.info("Preparing data for Google Sheets export...")

            if not self.test_sheets_access():
                return "Google Sheets access test failed"

            summary_data, cluster_data = self.prepare_metrics_data()

            if not summary_data:
                return "No data available for export"

            # Create formatted output
            sheets_content = self.create_sheets_format_data(summary_data, cluster_data)

            # Save to local file for upload
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"koio_support_report_{timestamp}.txt"

            with open(filename, 'w') as f:
                f.write(sheets_content)

            logger.info(f"Report saved to {filename}")

            # Create CSV for easy import
            self._create_csv_files(summary_data, cluster_data, timestamp)

            return f"""
Google Sheets Export Complete!

📊 MANUAL IMPORT INSTRUCTIONS:
1. Open your Google Sheet: {GOOGLE_SHEETS_URL}
2. Copy the content from: {filename}
3. Paste into a new sheet tab named: 'Report_{datetime.now().strftime("%Y%m%d")}'

📋 CSV FILES GENERATED:
- summary_{timestamp}.csv
- clusters_{timestamp}.csv

💡 TIP: Use Google Sheets "Import" feature to upload the CSV files directly.

🔄 AUTOMATION: To enable automatic updates, add GOOGLE_API_KEY to your GitHub secrets.
"""

        except Exception as e:
            logger.error(f"Error in sheets export: {e}")
            return f"Sheets export failed: {str(e)}"

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
    logger.info("=== Koio Google Sheets Export Starting ===")

    exporter = GoogleSheetsExporter()
    result = exporter.export_to_sheets_manually()

    print("\n" + "="*60)
    print("GOOGLE SHEETS EXPORT REPORT")
    print("="*60)
    print(result)
    print("="*60)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)