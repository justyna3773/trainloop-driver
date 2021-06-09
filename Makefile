VERSION_TAG="1.3.5"

docker:
	docker build -f Dockerfile -t amalota/trainloop-driver:${VERSION_TAG} .

docker-push: docker
	docker tag amalota/trainloop-driver:${VERSION_TAG} amalota/trainloop-driver:latest
	docker push amalota/trainloop-driver:${VERSION_TAG}
	docker push amalota/trainloop-driver:latest

test: docker
	docker-compose -f docker-compose-tests.yml down
	docker-compose -f docker-compose-tests.yml rm -f 
	docker-compose -f docker-compose-tests.yml up
