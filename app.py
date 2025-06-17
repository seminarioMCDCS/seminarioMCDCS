from flask import Flask, render_template
from webapp import app as flask_app
from starlette.middleware.wsgi import WSGIMiddleware
from fastapi import FastAPI

app = FastAPI()
app.mount("/", WSGIMiddleware(flask_app))