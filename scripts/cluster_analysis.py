import os
import time
import json
import sys
import traceback
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
import hdbscan
from sklearn.preprocessing import StandardScaler

load_dotenv()

# --- Environment Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Clustering Configuration
ANALYSIS_DAYS = int(os.getenv("CLUSTER_ANALYSIS_DAYS", "30"))
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
MAX_CLUSTERS = int(os.getenv("MAX_CLUSTERS", "15"))
MIN_NEW_TICKETS_FOR_RERUN = int(os.getenv("MIN_NEW_TICKETS_FOR_RERUN", "10"))
HDBSCAN_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "2"))
CLUSTER_SELECTION_EPSILON = float(os.getenv("CLUSTER_SELECTION_EPSILON", "0.1"))

# AI Configuration
GPT_MODEL = os.getenv("CLUSTER_GPT_MODEL", "gpt-4o-mini")
MAX_SUMMARY_TOKENS = int(os.getenv("MAX_SUMMARY_TOKENS", "300"))
BATCH_SIZE = int(os.getenv("CLUSTER_BATCH_SIZE", "50"))

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
class ClusterInfo:
    cluster_id: int
    size: int
    tickets: List[Dict[str, Any]]
    centroid: Optional[List[float]]
    theme_name: str = ""
    summary: str = ""
    severity: int = 1
    action_items: str = ""
    avg_sentiment: float = 0.0
    sentiment_trend: str = "neutral"

class ClusterAnalyzer:
    def __init__(self):
        self.tickets_data = []
        self.embeddings_matrix = None
        self.clusters = {}

    def fetch_tickets_with_embeddings(self) -> bool:
        """Fetch tickets from last N days that have embeddings."""
        cutoff_date = (datetime.now() - timedelta(days=ANALYSIS_DAYS)).isoformat()

        try:
            logger.info(f"Fetching tickets with embeddings from last {ANALYSIS_DAYS} days...")

            query = (
                sb.table("raw_gorgias")
                .select("id, subject, message_for_ai, ai_labels, ai_sentiment, ai_priority, ai_summary, embedding, created_datetime, updated_datetime")
                .gte("created_datetime", cutoff_date)
                .not_.is_("embedding", "null")
                .not_.is_("message_for_ai", "null")
                .order("created_datetime", desc=True)
            )

            result = query.execute()
            self.tickets_data = result.data or []

            logger.info(f"Found {len(self.tickets_data)} tickets with embeddings")

            if len(self.tickets_data) < MIN_CLUSTER_SIZE:
                logger.warning(f"Not enough tickets ({len(self.tickets_data)}) for clustering (min: {MIN_CLUSTER_SIZE})")
                return False

            return True

        except Exception as e:
            logger.error(f"Error fetching tickets: {e}")
            return False

    def should_run_clustering(self) -> bool:
        """Check if we should run clustering based on new tickets."""
        try:
            # Get last clustering run timestamp
            query = (
                sb.table("ticket_clusters")
                .select("created_at")
                .order("created_at", desc=True)
                .limit(1)
            )

            result = query.execute()

            if not result.data:
                logger.info("No previous clustering run found - running initial clustering")
                return True

            last_run = datetime.fromisoformat(result.data[0]['created_at'].replace('Z', '+00:00'))

            # Count new tickets since last run
            new_tickets_query = (
                sb.table("raw_gorgias")
                .select("id", count="exact")
                .gte("created_datetime", last_run.isoformat())
                .not_.is_("embedding", "null")
            )

            new_count_result = new_tickets_query.execute()
            new_ticket_count = new_count_result.count or 0

            logger.info(f"Found {new_ticket_count} new tickets since last clustering run")

            if new_ticket_count >= MIN_NEW_TICKETS_FOR_RERUN:
                logger.info(f"Running clustering - {new_ticket_count} new tickets >= {MIN_NEW_TICKETS_FOR_RERUN} threshold")
                return True
            else:
                logger.info(f"Skipping clustering - only {new_ticket_count} new tickets < {MIN_NEW_TICKETS_FOR_RERUN} threshold")
                return False

        except Exception as e:
            logger.error(f"Error checking clustering necessity: {e}")
            # Run clustering on error to be safe
            return True

    def prepare_embeddings_matrix(self) -> bool:
        """Convert embeddings to numpy matrix for clustering."""
        try:
            embeddings = []
            valid_tickets = []

            for ticket in self.tickets_data:
                if ticket.get('embedding'):
                    try:
                        # Handle both list and string formats
                        if isinstance(ticket['embedding'], str):
                            embedding = json.loads(ticket['embedding'])
                        else:
                            embedding = ticket['embedding']

                        if len(embedding) == 1536:  # Verify expected dimension
                            embeddings.append(embedding)
                            valid_tickets.append(ticket)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Invalid embedding for ticket {ticket['id']}: {e}")
                        continue

            if len(embeddings) < MIN_CLUSTER_SIZE:
                logger.error(f"Not enough valid embeddings: {len(embeddings)} < {MIN_CLUSTER_SIZE}")
                return False

            self.embeddings_matrix = np.array(embeddings)
            self.tickets_data = valid_tickets

            logger.info(f"Prepared embeddings matrix: {self.embeddings_matrix.shape}")
            return True

        except Exception as e:
            logger.error(f"Error preparing embeddings matrix: {e}")
            return False

    def perform_clustering(self) -> bool:
        """Run HDBSCAN clustering on embeddings."""
        try:
            logger.info("Starting HDBSCAN clustering...")

            # Normalize embeddings for better clustering
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(self.embeddings_matrix)

            # Configure HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
                metric='euclidean'
            )

            cluster_labels = clusterer.fit_predict(normalized_embeddings)

            # Process clustering results
            unique_labels = set(cluster_labels)
            noise_points = list(cluster_labels).count(-1)

            logger.info(f"Clustering complete: {len(unique_labels)} clusters, {noise_points} noise points")

            # Group tickets by cluster
            self.clusters = {}
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Skip noise points
                    continue

                if label not in self.clusters:
                    self.clusters[label] = {
                        'tickets': [],
                        'embeddings': []
                    }

                self.clusters[label]['tickets'].append(self.tickets_data[i])
                self.clusters[label]['embeddings'].append(self.embeddings_matrix[i])

            # Filter out small clusters and limit to MAX_CLUSTERS
            valid_clusters = {k: v for k, v in self.clusters.items()
                            if len(v['tickets']) >= MIN_CLUSTER_SIZE}

            if len(valid_clusters) > MAX_CLUSTERS:
                # Keep largest clusters
                sorted_clusters = sorted(valid_clusters.items(),
                                       key=lambda x: len(x[1]['tickets']),
                                       reverse=True)
                valid_clusters = dict(sorted_clusters[:MAX_CLUSTERS])
                logger.info(f"Limited to {MAX_CLUSTERS} largest clusters")

            self.clusters = valid_clusters
            logger.info(f"Final clusters: {len(self.clusters)} valid clusters")

            return len(self.clusters) > 0

        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            traceback.print_exc()
            return False

    def calculate_cluster_centroids(self):
        """Calculate centroid embeddings for each cluster."""
        for cluster_id, cluster_data in self.clusters.items():
            embeddings = np.array(cluster_data['embeddings'])
            centroid = np.mean(embeddings, axis=0).tolist()
            cluster_data['centroid'] = centroid

    def generate_cluster_summaries(self) -> bool:
        """Use GPT-4o-mini to generate business summaries for each cluster."""
        try:
            logger.info("Generating AI summaries for clusters...")

            for cluster_id, cluster_data in self.clusters.items():
                tickets = cluster_data['tickets']

                # Prepare sample messages for analysis
                sample_messages = []
                sentiments = []
                priorities = []

                for ticket in tickets[:10]:  # Analyze up to 10 representative tickets
                    if ticket.get('message_for_ai'):
                        sample_messages.append(f"Subject: {ticket.get('subject', '')}\nMessage: {ticket['message_for_ai'][:500]}")

                    if ticket.get('ai_sentiment'):
                        sentiments.append(ticket['ai_sentiment'])
                    if ticket.get('ai_priority'):
                        priorities.append(ticket['ai_priority'])

                # Calculate cluster metrics
                avg_sentiment = self._calculate_sentiment_score(sentiments)
                avg_priority = np.mean(priorities) if priorities else 3

                # Generate summary with GPT
                summary_prompt = f"""Analyze this cluster of {len(tickets)} customer support tickets and provide:

1. Theme Name: 2-4 word category (e.g., "Sizing Issues", "Shipping Delays")
2. Summary: 2-3 sentences describing the common theme
3. Severity: Integer 1 (low impact) to 5 (critical business issue)
4. Action Items: Specific actionable recommendations

Sample tickets from this cluster:
{chr(10).join(sample_messages[:5])}

Respond as JSON with keys: theme_name, summary, severity, action_items"""

                try:
                    response = openai_client.chat.completions.create(
                        model=GPT_MODEL,
                        messages=[{"role": "user", "content": summary_prompt}],
                        max_tokens=MAX_SUMMARY_TOKENS,
                        temperature=0.3
                    )

                    response_content = response.choices[0].message.content.strip()
                    logger.info(f"GPT response for cluster {cluster_id}: {response_content[:100]}...")

                    # Handle potential markdown code blocks
                    if response_content.startswith('```'):
                        # Extract JSON from code block
                        lines = response_content.split('\n')
                        json_lines = []
                        in_json = False
                        for line in lines:
                            if line.strip().startswith('```'):
                                in_json = not in_json
                                continue
                            if in_json:
                                json_lines.append(line)
                        response_content = '\n'.join(json_lines)

                    summary_data = json.loads(response_content)

                    cluster_data.update({
                        'theme_name': summary_data.get('theme_name', f'Cluster {cluster_id}'),
                        'summary': summary_data.get('summary', ''),
                        'severity': min(5, max(1, int(summary_data.get('severity', 3)))),
                        'action_items': summary_data.get('action_items', ''),
                        'avg_sentiment': avg_sentiment,
                        'sentiment_trend': self._sentiment_to_trend(avg_sentiment),
                        'size': len(tickets)
                    })

                    logger.info(f"Generated summary for cluster {cluster_id}: {cluster_data['theme_name']}")
                    time.sleep(0.5)  # Rate limiting

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.warning(f"Failed to parse GPT response for cluster {cluster_id}: {e}")
                    logger.warning(f"Raw response: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
                    cluster_data.update({
                        'theme_name': f'Cluster {cluster_id}',
                        'summary': 'AI summary generation failed',
                        'severity': 3,
                        'action_items': 'Manual review required',
                        'avg_sentiment': avg_sentiment,
                        'sentiment_trend': self._sentiment_to_trend(avg_sentiment),
                        'size': len(tickets)
                    })

            return True

        except Exception as e:
            logger.error(f"Error generating cluster summaries: {e}")
            return False

    def _calculate_sentiment_score(self, sentiments: List[str]) -> float:
        """Convert sentiment labels to numeric score."""
        if not sentiments:
            return 0.0

        score_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
        scores = [score_map.get(s.lower(), 0.0) for s in sentiments]
        return np.mean(scores)

    def _sentiment_to_trend(self, score: float) -> str:
        """Convert sentiment score to trend label."""
        if score > 0.3:
            return 'positive'
        elif score < -0.3:
            return 'negative'
        else:
            return 'neutral'

    def save_clustering_results(self) -> bool:
        """Save clusters and assignments to database."""
        try:
            logger.info("Saving clustering results to database...")

            current_time = datetime.now().isoformat()

            # Save cluster metadata
            for cluster_id, cluster_data in self.clusters.items():
                cluster_record = {
                    'cluster_id': int(cluster_id),
                    'theme_name': cluster_data['theme_name'],
                    'summary': cluster_data['summary'],
                    'severity': int(cluster_data['severity']),
                    'action_items': cluster_data['action_items'],
                    'ticket_count': int(cluster_data['size']),
                    'avg_sentiment': float(cluster_data['avg_sentiment']),
                    'sentiment_trend': cluster_data['sentiment_trend'],
                    'centroid_embedding': json.dumps(cluster_data['centroid']),
                    'created_at': current_time
                }

                sb.table("ticket_clusters").insert(cluster_record).execute()

                # Save ticket assignments
                assignments = []
                for ticket in cluster_data['tickets']:
                    assignments.append({
                        'ticket_id': int(ticket['id']),
                        'cluster_id': int(cluster_id),
                        'assigned_at': current_time
                    })

                if assignments:
                    sb.table("ticket_cluster_assignments").insert(assignments).execute()

            # Save trend snapshot
            trend_snapshot = {
                'analysis_date': current_time,
                'total_tickets': len(self.tickets_data),
                'total_clusters': len(self.clusters),
                'analysis_period_days': ANALYSIS_DAYS,
                'cluster_distribution': json.dumps({
                    str(cid): cdata['size'] for cid, cdata in self.clusters.items()
                })
            }

            sb.table("cluster_trends").insert(trend_snapshot).execute()

            logger.info(f"Successfully saved {len(self.clusters)} clusters to database")
            return True

        except Exception as e:
            logger.error(f"Error saving clustering results: {e}")
            traceback.print_exc()
            return False

    def run_analysis(self) -> bool:
        """Main analysis pipeline."""
        try:
            logger.info("Starting cluster analysis pipeline...")

            # Step 1: Fetch data
            if not self.fetch_tickets_with_embeddings():
                logger.error("Failed to fetch tickets with embeddings")
                return False

            # Step 2: Check if clustering is needed
            if not self.should_run_clustering():
                logger.info("Clustering not needed - exiting")
                return True

            # Step 3: Prepare embeddings
            if not self.prepare_embeddings_matrix():
                logger.error("Failed to prepare embeddings matrix")
                return False

            # Step 4: Perform clustering
            if not self.perform_clustering():
                logger.error("Clustering failed or produced no valid clusters")
                return False

            # Step 5: Calculate centroids
            self.calculate_cluster_centroids()

            # Step 6: Generate AI summaries
            if not self.generate_cluster_summaries():
                logger.error("Failed to generate cluster summaries")
                return False

            # Step 7: Save results
            if not self.save_clustering_results():
                logger.error("Failed to save clustering results")
                return False

            logger.info("Cluster analysis pipeline completed successfully")

            # Log summary
            for cluster_id, cluster_data in self.clusters.items():
                logger.info(f"Cluster {cluster_id}: {cluster_data['theme_name']} "
                          f"({cluster_data['size']} tickets, severity {cluster_data['severity']})")

            return True

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            traceback.print_exc()
            return False

def main():
    logger.info("=== Koio Cluster Analysis Starting ===")

    analyzer = ClusterAnalyzer()
    success = analyzer.run_analysis()

    if success:
        logger.info("=== Cluster Analysis Completed Successfully ===")
        sys.exit(0)
    else:
        logger.error("=== Cluster Analysis Failed ===")
        sys.exit(1)

if __name__ == "__main__":
    main()