flwr run server:server.app --run-config config.yaml

flwr run client:client.app --config config.yaml --node-config='{"client_id": 0}'


flwr run client:client_app.app \
  --config config.yaml \
  --node-config='{"client_id": 1}'
