# firestore_bridge.py
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os

class DarwinianBridge:
    def __init__(self, project_id="stocktrader-378901", db_name="darwinian-v5"):
        # Initialize connection if not already active
        if not firebase_admin._apps:
            # Use Application Default Credentials (works on Cloud Studio & Cloud Run)
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {
                'projectId': project_id,
            })

        # Connect specifically to the Sidecar DB
        try:
            self.db = firestore.client(database_id=db_name)
            self.portfolio_ref = self.db.collection('portfolio').document('main')
            self.system_ref = self.db.collection('system_state').document('latest')
        except Exception as e:
            print(f"⚠️ Bridge Connection Failed: {e}")
            self.db = None

    def update_portfolio(self, account_data):
        """
        Updates the main portfolio document and records history.
        Expects account_data to be an object (like Alpaca Account) with .equity and .cash
        """
        if not self.db: return

        equity = float(account_data.equity)
        cash = float(account_data.cash)

        # 1. Update the "Snapshot" (What the Dashboard sees right now)
        self.portfolio_ref.set({
            'equity': equity,
            'cash': cash,
            'last_updated': firestore.SERVER_TIMESTAMP
        }, merge=True)

        # 2. Add to History (For the Graph)
        self.db.collection('equity_history').add({
            'equity': equity,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        print(f"✅ Bridge: Portfolio updated (${equity:,.2f})")

    def update_positions(self, positions):
        """
        Overwrites the positions sub-collection with current real data.
        Expects a list of Alpaca Position objects.
        """
        if not self.db: return

        batch = self.db.batch()
        positions_col = self.portfolio_ref.collection('positions')

        # OPTIONAL: Delete old positions first (to remove sold stocks)
        # For high-frequency, it's better to just update, but for accuracy, we wipe and rewrite.
        old_docs = positions_col.list_documents()
        for doc in old_docs:
            batch.delete(doc)

        # Add current positions
        for pos in positions:
            doc_ref = positions_col.document(pos.symbol)
            batch.set(doc_ref, {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_profit_loss': float(pos.unrealized_pl),
                'last_updated': firestore.SERVER_TIMESTAMP
            })

        batch.commit()
        print(f"✅ Bridge: Synced {len(positions)} positions.")

    def log_action(self, message, action_type="INFO"):
        """
        Sends an audit log to the dashboard feed.
        """
        if not self.db: return

        self.db.collection('audit_logs').add({
            'timestamp': firestore.SERVER_TIMESTAMP,
            'action': action_type,
            'message': message,
            'weekend_recap': message # Redundant save for safety
        })
