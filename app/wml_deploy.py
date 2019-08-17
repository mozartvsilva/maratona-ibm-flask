from watson_machine_learning_client import WatsonMachineLearningAPIClient

credentials	= {
  "apikey": "xQW-tHywRRnkx2RXEOvBwY7a477Te1SyszEwITAgqA0I",
  "instance_id": "f08120b8-fdb6-43c5-96a8-38f71bc1e643",
  "url": "https://us-south.ml.cloud.ibm.com",
}
client = WatsonMachineLearningAPIClient(credentials)
metadata = {
    client.repository.ModelMetaNames.NAME: "keras model",
    client.repository.ModelMetaNames.FRAMEWORK_NAME: "tensorflow",
    client.repository.ModelMetaNames.FRAMEWORK_VERSION: "1.13",
    client.repository.ModelMetaNames.FRAMEWORK_LIBRARIES: [{'name':'keras', 'version': '2.1.6'}]
}
model_details = client.repository.store_model( model="model.tgz", meta_props=metadata)

# model_id = model_details["metadata"]["guid"]
# model_deployment_details = client.deployments.create( artifact_uid=model_id, name="My Keras model deployment" )