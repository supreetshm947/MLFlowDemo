start:
	echo "Starting MLFlow server"
	mlflow server --host 127.0.0.1 --port 8080
install_minikube:
	echo "Installing Minikube"
	curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
	sudo install minikube-linux-amd64 /usr/local/bin/minikube
	sudo rm -f minikube-linux-amd64
install_kubectl:
	echo "Installing Kubectl"
	sudo apt-get install -y apt-transport-https ca-certificates
	sudo mkdir -p /etc/apt/keyrings
	curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
	sudo chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg
	echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
	sudo chmod 644 /etc/apt/sources.list.d/kubernetes.list
	sudo apt update
	sudo apt-get install -y kubectl
setup:
	install_minikube install_kubectl
start_minikube:
	minikube start --driver=docker
push_image:
	docker tag deployment_image:latest supreetsharma94/deployment_image:latest
	docker push supreetsharma94/deployment_image:latest
setup_cluster:
	kubectl apply -f deployment.yaml
	kubectl apply -f service.yaml