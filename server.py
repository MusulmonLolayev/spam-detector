from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from models import SpamDetector
from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

templates = Jinja2Templates(directory='templates')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post('/detect')
def blog(text: str):
    model = SpamDetector.load()
    
    y = model.predict([text])
    y *= 100
    res = {'non_spam': int(y[0, 0]),
           'spam': int(y[0, 1])}
    return res

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8080)