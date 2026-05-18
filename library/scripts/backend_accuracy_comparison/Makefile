.PHONY: help setup check test-numerical test-closed-loop clean

help:
	@echo "Backend Accuracy Comparison - Make Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup              - Install all dependencies"
	@echo "  make check              - Check if environment is ready"
	@echo "  make test-numerical     - Run quick numerical comparison with PI05 on Libero Spatial dataset"
	@echo "  make test-closed-loop   - Run quick closed-loop benchmark with PI05 on Libero 10 task suite"
	@echo "  make clean              - Remove generated files"
	@echo ""
	@echo "For full documentation see README.md"

setup:
	@echo "Installing dependencies..."
	@echo ""
	@echo "Note: For development setup, run from library/ root:"
	@echo "  cd ../../ && uv sync --all-extras"
	@echo ""
	@echo "Installing additional dependencies..."
	uv pip install hf-libero
	@echo ""
	@echo "✓ Dependencies installed"
	@echo "Run 'make check' to verify"

check:
	@echo "Checking environment..."
	python check_setup.py

test-numerical:
	@echo "Running numerical comparison (quick test)..."
	@echo "This will use a pretrained model from HuggingFace"
	@echo ""
	@mkdir -p results exports checkpoints
	@echo "Downloading and preparing checkpoint..."
	python -c "from physicalai.policies.lerobot import PI05; p = PI05.from_pretrained('lerobot/pi05_libero_finetuned_v044'); p.save('checkpoints/test_pi05.ckpt')"
	python numerical_comparison.py \
		--checkpoint checkpoints/test_pi05.ckpt \
		--policy-class physicalai.policies.pi05.Pi05 \
		--dataset lerobot/libero_10_image \
		--num-samples 50 \
		--output results/test_numerical.json
	@echo ""
	@echo "✓ Numerical test complete"
	@echo "Results: results/test_numerical.json"

test-closed-loop:
	@echo "Running closed-loop benchmark (quick test)..."
	@echo "This may take 5-10 minutes..."
	@echo ""
	@mkdir -p results exports checkpoints
	@if [ ! -f checkpoints/test_pi05.ckpt ]; then \
		echo "Downloading and preparing checkpoint..."; \
		python -c "from physicalai.policies.lerobot import PI05; p = PI05.from_pretrained('lerobot/pi05_libero_finetuned_v044'); p.save('checkpoints/test_pi05.ckpt')"; \
	fi
	python closed_loop_benchmark.py \
		--checkpoint checkpoints/test_pi05.ckpt \
		--policy-class physicalai.policies.lerobot.PI05 \
		--task-suite libero_10 \
		--task-ids 0 1 \
		--num-episodes 5 \
		--output results/test_closed_loop.json
	@echo ""
	@echo "✓ Closed-loop test complete"
	@echo "Results: results/test_closed_loop.json"

clean:
	@echo "Cleaning generated files..."
	rm -rf results/ exports/ checkpoints/test_pi05.ckpt
	rm -rf __pycache__/ *.pyc
	@echo "✓ Cleaned"
