# ========================
# Makefile for ProspectIQ AI Dataset Generator
# ========================

IMAGE_NAME=prospectiq-generator
CONTAINER_NAME=prospectiq-run
OUTPUT_DIR=./prospectiq_out

# Build Docker image
build:
	docker build -t $(IMAGE_NAME)

# Run generator inside container
# Pass OPENAI_API_KEY from host environment
run:
	docker run --rm \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-v $(CURDIR):/app \
		$(IMAGE_NAME) \
		--companies 8 --reps 10 --seed 42 --model gpt-4o-mini --lang en --outdir $(OUTPUT_DIR)

# Shell inside container (for debugging)
shell:
	docker run -it --rm \
		-e OPENAI_API_KEY=$$OPENAI_API_KEY \
		-v $(CURDIR):/app \
		$(IMAGE_NAME) /bin/bash
		
