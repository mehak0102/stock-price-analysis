import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Default SQLite database
DEFAULT_DATABASE_URL = "sqlite:///stock_app.db"

# Environment database
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False}
)

# Base class
Base = declarative_base()


# -------------------------------
# User Table
# -------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<User(username='{self.username}')>"


# -------------------------------
# Watchlist Table
# -------------------------------
class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    stock_symbol = Column(String, nullable=False)
    added_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<WatchlistItem(user_id={self.user_id}, stock_symbol='{self.stock_symbol}')>"


# -------------------------------
# Prediction Table
# -------------------------------
class StockPrediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    stock_symbol = Column(String, nullable=False)
    prediction_date = Column(DateTime, default=datetime.datetime.utcnow)
    target_date = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)
    model_type = Column(String, nullable=False)

    def __repr__(self):
        return f"<StockPrediction(stock_symbol='{self.stock_symbol}', target_date='{self.target_date}')>"


# -------------------------------
# Initialize DB
# -------------------------------
def initialize_database():
    Base.metadata.create_all(engine)
    print("Database initialized.")


# -------------------------------
# Session
# -------------------------------
def get_session():
    Session = sessionmaker(bind=engine)
    return Session()


# -------------------------------
# Add User
# -------------------------------
def add_user(username):
    session = get_session()
    try:
        user = User(username=username)
        session.add(user)
        session.commit()
        return user.id
    except Exception as e:
        session.rollback()
        st.error(f"Error adding user: {e}")
        return None
    finally:
        session.close()


# -------------------------------
# Get User
# -------------------------------
def get_user_by_username(username):
    session = get_session()
    try:
        return session.query(User).filter_by(username=username).first()
    except Exception as e:
        st.error(f"Error getting user: {e}")
        return None
    finally:
        session.close()


# -------------------------------
# Add to Watchlist
# -------------------------------
def add_to_watchlist(user_id, stock_symbol):
    session = get_session()
    try:
        existing = session.query(WatchlistItem).filter_by(
            user_id=user_id,
            stock_symbol=stock_symbol
        ).first()

        if existing:
            return False

        item = WatchlistItem(user_id=user_id, stock_symbol=stock_symbol)
        session.add(item)
        session.commit()
        return True

    except Exception as e:
        session.rollback()
        st.error(f"Error adding to watchlist: {e}")
        return False

    finally:
        session.close()


# -------------------------------
# Remove from Watchlist
# -------------------------------
def remove_from_watchlist(user_id, stock_symbol):
    session = get_session()

    try:
        item = session.query(WatchlistItem).filter_by(
            user_id=user_id,
            stock_symbol=stock_symbol
        ).first()

        if item:
            session.delete(item)
            session.commit()
            return True

        return False

    except Exception as e:
        session.rollback()
        st.error(f"Error removing from watchlist: {e}")
        return False

    finally:
        session.close()


# -------------------------------
# Get Watchlist
# -------------------------------
def get_watchlist(user_id):
    session = get_session()

    try:
        watchlist = session.query(WatchlistItem).filter_by(user_id=user_id).all()
        return [item.stock_symbol for item in watchlist]

    except Exception as e:
        st.error(f"Error getting watchlist: {e}")
        return []

    finally:
        session.close()


# -------------------------------
# Save Prediction
# -------------------------------
def save_prediction(stock_symbol, target_date, predicted_price, model_type, user_id=None):

    session = get_session()

    try:
        prediction = StockPrediction(
            user_id=user_id,
            stock_symbol=stock_symbol,
            target_date=target_date,
            predicted_price=predicted_price,
            model_type=model_type
        )

        session.add(prediction)
        session.commit()

        return prediction.id

    except Exception as e:
        session.rollback()
        st.error(f"Error saving prediction: {e}")
        return None

    finally:
        session.close()


# -------------------------------
# Update Prediction
# -------------------------------
def update_prediction_actual_price(prediction_id, actual_price):

    session = get_session()

    try:
        prediction = session.query(StockPrediction).filter_by(id=prediction_id).first()

        if prediction:
            prediction.actual_price = actual_price
            session.commit()
            return True

        return False

    except Exception as e:
        session.rollback()
        st.error(f"Error updating prediction: {e}")
        return False

    finally:
        session.close()


# -------------------------------
# Get Predictions
# -------------------------------
def get_user_predictions(user_id=None, limit=10):

    session = get_session()

    try:
        query = session.query(StockPrediction)

        if user_id:
            query = query.filter_by(user_id=user_id)

        predictions = query.order_by(
            StockPrediction.prediction_date.desc()
        ).limit(limit).all()

        data = []

        for p in predictions:
            data.append({
                "id": p.id,
                "stock_symbol": p.stock_symbol,
                "prediction_date": p.prediction_date,
                "target_date": p.target_date,
                "predicted_price": p.predicted_price,
                "actual_price": p.actual_price,
                "model_type": p.model_type
            })

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Error getting predictions: {e}")
        return pd.DataFrame()

    finally:
        session.close()


# -------------------------------
# Prediction Accuracy
# -------------------------------
def get_prediction_accuracy(user_id=None):

    session = get_session()

    try:
        query = session.query(StockPrediction).filter(
            StockPrediction.actual_price.isnot(None)
        )

        if user_id:
            query = query.filter_by(user_id=user_id)

        predictions = query.all()

        if not predictions:
            return {
                "total_predictions": 0,
                "avg_accuracy": 0,
                "correct_direction": 0
            }

        total = len(predictions)
        accuracy_sum = 0

        for p in predictions:
            accuracy = 100 - abs(
                (p.predicted_price - p.actual_price) / p.actual_price * 100
            )
            accuracy_sum += accuracy

        return {
            "total_predictions": total,
            "avg_accuracy": accuracy_sum / total
        }

    except Exception as e:
        st.error(f"Error calculating accuracy: {e}")
        return {
            "total_predictions": 0,
            "avg_accuracy": 0
        }

    finally:
        session.close()


# -------------------------------
# Run directly
# -------------------------------
if __name__ == "__main__":
    initialize_database()