import uvicorn
from app.config import settings # Import the settings instance

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app", # Path to the app instance
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
