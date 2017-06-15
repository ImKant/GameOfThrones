#!/usr/bin/env python
#-*-coding=utf-8-*-

from flask import Flask

App = Flask(__name__)

@App.route("/")
def index():
    return "Hello World"
