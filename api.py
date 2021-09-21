from colabcode import ColabCode
server = ColabCode(port=10000, code=False)
server.run_app(app=app)