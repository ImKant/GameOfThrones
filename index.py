#!/usr/bin/python
from flup.server.fcgi import WSGIServer
from flask import Flask, request
#from bson.objectid import ObjectId
#import Encrypt


def myapp(environ, start_response):

        start_response('200 OK', [('Content-Type', 'text/plain')])
def myapp(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    (rquest_method,data) = get_environ(environ)
    if rquest_method=='GET':
            return 'Get QueryStr:'+data
    else:
            return 'POST RequestData'+data


def get_environ(environ):
    rquest_method = environ["REQUEST_METHOD"]
    data=''
    if rquest_method=='GET':
            data=environ["QUERY_STRING"]
    else:
            data = environ["wsgi.input"].read()

    return (rquest_method,data)


if __name__ == '__main__':
    WSGIServer(myapp, bindAddress=('127.0.0.1',8008)).run()
