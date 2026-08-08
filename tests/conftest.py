import os
import pytest

# Ensure test DB path is set before any app/db imports
os.environ["REXA_DB_PATH"] = "web/rexa_test.db"

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    from web.db import init_db
    db_path = "web/rexa_test.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    init_db()
    yield
    
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
