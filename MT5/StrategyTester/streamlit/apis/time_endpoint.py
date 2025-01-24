# api/main.py
from fastapi import FastAPI, HTTPException
from datetime import datetime
from pydantic import BaseModel
import pytz

class TimeResponse(BaseModel):
    timestamp: str
    utc_time: str
    local_time: str

app = FastAPI(
    title="MT5 Strategy API",
    description="API for MT5 Strategy Tester",
    version="1.0.0"
)

# @app.get("/api/time", response_model=TimeResponse)
@app.get("/", response_model=TimeResponse)
async def get_current_time():
    try:
        now = datetime.now()
        utc_now = datetime.now(pytz.UTC)
        
        return TimeResponse(
            timestamp=str(int(now.timestamp())),
            utc_time=utc_now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            local_time=now.strftime("%Y-%m-%d %H:%M:%S Local")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=80)
    uvicorn.run(app, host="127.0.0.1", port=80)