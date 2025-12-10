from slack_api import SlackBot as ExternalSlackBot

# Reexportamos para mantener la interfaz usada en slack_handler.py
SlackBot = ExternalSlackBot
