Help me create a topology visualizer serrvice with OSMx using python

I want to be able to have an api that takes in the mininet setup python file testnet.py in my case and it convert it to an image that i can show in grafana in topology dashboard

The api should use flask

The app will run in a container and it should have a dockerfile and a requirements.txt
 @docker_setup/grafana/dashboards/topology.json  @docker_setup/services/topology/Dockerfile  @docker_setup/docker-compose.yml  @app/scenario_notification/testnet.py