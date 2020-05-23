import http.server
import socketserver
import os

PORT = 6001

# change directory to root node as we use absolute paths in html images
os.chdir('/')

Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), Handler)
print("serving at port", PORT)
httpd.serve_forever()
