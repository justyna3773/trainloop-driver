VERSION_TAG="1.0"

docker:
	docker build -f Dockerfile -t pkoperek/trainloop-driver:${VERSION_TAG} .

docker-push: docker
	docker tag pkoperek/trainloop-driver:${VERSION_TAG} pkoperek/trainloop-driver:latest
	docker push pkoperek/trainloop-driver:${VERSION_TAG} 
	docker push pkoperek/trainloop-driver:latest

test: docker
	docker-compose down
	docker-compose rm -f 
	docker-compose up
