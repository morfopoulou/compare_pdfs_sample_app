## Sample Compare PDFs with Gemini Application

## Clone

git clone https://github.com/morfopoulou/compare_pdfs_sample_app.git

## Deploy to Cloud Run

Ensure the default Cloud Run service account has the following IAM permissions:
- Cloud Build Editor
- Logging Admin
- Storage Object User
- Vertex AI User
- Artifact Registry Administrator


Set the environment variables in `initial.sh'
Add your app name in the Dockerfile
```In Cloud Shell, execute the following commands:
gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"
```
Before you can push or pull images, configure Docker to use the Google Cloud CLI to authenticate requests to Artifact Registry.
In Cloud Shell, execute the following command to create a registry

```
gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker
```
```
To set up authentication to Docker repositories in the region us-west1, run the following command:
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
```
```
To build the Docker image of your application and push it to Google Artifact Registry (AR)
gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"
```

``` Now deploy in Cloud Run
gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --region=$GCP_REGION \
  --platform=managed  \
  --project=$GCP_PROJECT \
  --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION,GCP_BUCKET=$GCP_BUCKET
```
