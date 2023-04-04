gzip -d gt/*.jsonl.gz
gzip -d sample_output.jsonl.gz
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/train.tgz .
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/validation.tgz .
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/test.tgz .
tar -xzvf train.tgz
tar -xzvf validation.tgz
tar -xzvf test.tgz