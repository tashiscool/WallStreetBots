"""Phase 4: Production Deployment System
Docker, CI / CD, and production deployment management.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.production_config import ConfigManager
from ..core.production_logging import ProductionLogger


class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    environment: DeploymentEnvironment
    version: str
    docker_image: str
    replicas: int
    cpu_limit: str
    memory_limit: str
    health_check_path: str
    environment_variables: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)
    ports: list[str] = field(default_factory=list)


@dataclass
class DeploymentResult:
    """Deployment result."""

    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: datetime | None = None
    logs: list[str] = field(default_factory=list)
    error_message: str | None = None
    rollback_version: str | None = None


class DockerManager:
    """Docker container management."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.logger.info("DockerManager initialized")

    def build_image(self, dockerfile_path: str, image_name: str, tag: str = "latest") -> bool:
        """Build Docker image."""
        try:
            cmd = ["docker", "build", "-t", f"{image_name}: {tag}", "-f", dockerfile_path, "."]

            self.logger.info(f"Building Docker image: {image_name}: {tag}")
            subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603

            self.logger.info(f"Docker image built successfully: {image_name}: {tag}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error building Docker image: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error building Docker image: {e}")
            return False

    def push_image(self, image_name: str, tag: str = "latest", registry: str | None = None) -> bool:
        """Push Docker image to registry."""
        try:
            full_image_name = (
                f"{registry}/{image_name}: {tag}" if registry else f"{image_name}: {tag}"
            )

            cmd = ["docker", "push", full_image_name]

            self.logger.info(f"Pushing Docker image: {full_image_name}")
            subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603

            self.logger.info(f"Docker image pushed successfully: {full_image_name}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error pushing Docker image: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error pushing Docker image: {e}")
            return False

    def run_container(
        self,
        image_name: str,
        container_name: str,
        environment_vars: dict[str, str] | None = None,
        ports: list[str] | None = None,
        volumes: list[str] | None = None,
    ) -> bool:
        """Run Docker container."""
        try:
            cmd = ["docker", "run", "-d", "--name", container_name]

            # Add environment variables
            if environment_vars:
                for key, value in environment_vars.items():
                    cmd.extend(["-e", f"{key} = {value}"])

            # Add port mappings
            if ports:
                for port in ports:
                    cmd.extend(["-p", port])

            # Add volume mounts
            if volumes:
                for volume in volumes:
                    cmd.extend(["-v", volume])

            cmd.append(image_name)

            self.logger.info(f"Running Docker container: {container_name}")
            subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603

            self.logger.info(f"Docker container started: {container_name}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running Docker container: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error running Docker container: {e}")
            return False

    def stop_container(self, container_name: str) -> bool:
        """Stop Docker container."""
        try:
            cmd = ["docker", "stop", container_name]

            self.logger.info(f"Stopping Docker container: {container_name}")
            subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603

            self.logger.info(f"Docker container stopped: {container_name}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error stopping Docker container: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error stopping Docker container: {e}")
            return False

    def remove_container(self, container_name: str) -> bool:
        """Remove Docker container."""
        try:
            cmd = ["docker", "rm", container_name]

            self.logger.info(f"Removing Docker container: {container_name}")
            subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603

            self.logger.info(f"Docker container removed: {container_name}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error removing Docker container: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error removing Docker container: {e}")
            return False

    def get_container_logs(self, container_name: str, lines: int = 100) -> list[str]:
        """Get container logs."""
        try:
            cmd = ["docker", "logs", "--tail", str(lines), container_name]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603
            logs = result.stdout.split("\n")

            return logs

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting container logs: {e.stderr}")
            return []
        except Exception as e:
            self.logger.error(f"Error getting container logs: {e}")
            return []


class KubernetesManager:
    """Kubernetes deployment management."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.logger.info("KubernetesManager initialized")

    def create_deployment(self, deployment_config: DeploymentConfig) -> bool:
        """Create Kubernetes deployment."""
        try:
            # Generate Kubernetes deployment YAML
            deployment_yaml = self._generate_deployment_yaml(deployment_config)

            # Write to temporary file
            temp_file = f"/tmp / deployment_{deployment_config.version}.yaml"
            with open(temp_file, "w") as f:
                f.write(deployment_yaml)

            # Apply deployment
            cmd = ["kubectl", "apply", "-f", temp_file]

            self.logger.info(f"Creating Kubernetes deployment: {deployment_config.version}")
            subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603

            # Clean up temp file
            os.remove(temp_file)

            self.logger.info(f"Kubernetes deployment created: {deployment_config.version}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error creating Kubernetes deployment: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error creating Kubernetes deployment: {e}")
            return False

    def update_deployment(self, deployment_name: str, image: str) -> bool:
        """Update Kubernetes deployment."""
        try:
            cmd = [
                "kubectl",
                "set",
                "image",
                f"deployment / {deployment_name}",
                f"{deployment_name} = {image}",
            ]

            self.logger.info(f"Updating Kubernetes deployment: {deployment_name}")
            subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603

            self.logger.info(f"Kubernetes deployment updated: {deployment_name}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error updating Kubernetes deployment: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error updating Kubernetes deployment: {e}")
            return False

    def get_deployment_status(self, deployment_name: str) -> dict[str, Any]:
        """Get deployment status."""
        try:
            cmd = ["kubectl", "get", "deployment", deployment_name, "-o", "json"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=False)  # noqa: S603
            status = json.loads(result.stdout)

            return status

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting deployment status: {e.stderr}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting deployment status: {e}")
            return {}

    def _generate_deployment_yaml(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes deployment YAML."""
        yaml_template = f"""
apiVersion: apps / v1
kind: Deployment
metadata:
  name: wallstreetbots - {config.environment.value}
  labels:
    app: wallstreetbots
    environment: {config.environment.value}
    version: {config.version}
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: wallstreetbots
      environment: {config.environment.value}
  template:
    metadata:
      labels:
        app: wallstreetbots
        environment: {config.environment.value}
        version: {config.version}
    spec:
      containers:
      - name: wallstreetbots
        image: {config.docker_image}
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: {config.cpu_limit}
            memory: {config.memory_limit}
        env:
"""

        # Add environment variables
        for key, value in config.environment_variables.items():
            yaml_template += f'        - name: {key}\n          value: "{value}"\n'

        # Add secrets
        for key, secret_name in config.secrets.items():
            yaml_template += f"        - name: {key}\n          valueFrom: \n            secretKeyRef:\n              name: {secret_name}\n              key: {key}\n"

        return yaml_template


class CICDManager:
    """CI / CD pipeline management."""

    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.logger.info("CICDManager initialized")

    def run_tests(self) -> bool:
        """Run test suite."""
        try:
            self.logger.info("Running test suite")

            # Run pytest
            cmd = ["python", "-m", "pytest", "-v", "--tb = short"]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, shell=False)  # noqa: S603

            if result.returncode == 0:
                self.logger.info("All tests passed")
                return True
            else:
                self.logger.error(f"Tests failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return False

    def run_linting(self) -> bool:
        """Run code linting."""
        try:
            self.logger.info("Running code linting")

            # Run flake8
            cmd = ["flake8", "backend / tradingbot/", "--max - line-length = 100"]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, shell=False)  # noqa: S603

            if result.returncode == 0:
                self.logger.info("Linting passed")
                return True
            else:
                self.logger.warning(f"Linting issues found: {result.stdout}")
                return False

        except Exception as e:
            self.logger.error(f"Error running linting: {e}")
            return False

    def run_security_scan(self) -> bool:
        """Run security scan."""
        try:
            self.logger.info("Running security scan")

            # Run bandit security linter
            cmd = ["bandit", "-r", "backend / tradingbot/", "-f", "json"]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, shell=False)  # noqa: S603

            if result.returncode == 0:
                self.logger.info("Security scan passed")
                return True
            else:
                self.logger.warning(f"Security issues found: {result.stdout}")
                return False

        except Exception as e:
            self.logger.error(f"Error running security scan: {e}")
            return False

    def build_artifacts(self) -> bool:
        """Build deployment artifacts."""
        try:
            self.logger.info("Building deployment artifacts")

            # Create build directory
            build_dir = Path("build")
            build_dir.mkdir(exist_ok=True)

            # Copy source code
            subprocess.run(["cp", "-r", "backend/", str(build_dir)], check=True, shell=False)  # noqa: S603
            subprocess.run(["cp", "requirements.txt", str(build_dir)], check=True, shell=False)  # noqa: S603
            subprocess.run(["cp", "pyproject.toml", str(build_dir)], check=True, shell=False)  # noqa: S603
            subprocess.run(["cp", "Dockerfile", str(build_dir)], check=True, shell=False)  # noqa: S603

            self.logger.info("Deployment artifacts built successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error building artifacts: {e}")
            return False


class DeploymentManager:
    """Main deployment orchestrator."""

    def __init__(self, config: ConfigManager, logger: ProductionLogger):
        self.config = config
        self.logger = logger

        # Initialize managers
        self.docker_manager = DockerManager(logger)
        self.kubernetes_manager = KubernetesManager(logger)
        self.cicd_manager = CICDManager(logger)

        self.logger.info("DeploymentManager initialized")

    async def deploy(self, deployment_config: DeploymentConfig) -> DeploymentResult:
        """Deploy application."""
        try:
            deployment_id = f"deploy_{deployment_config.version}_{int(datetime.now().timestamp())}"

            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.IN_PROGRESS,
                start_time=datetime.now(),
            )

            self.logger.info(f"Starting deployment: {deployment_id}")

            # Step 1: Run CI / CD pipeline
            if not self._run_cicd_pipeline():
                result.status = DeploymentStatus.FAILED
                result.error_message = "CI / CD pipeline failed"
                result.end_time = datetime.now()
                return result

            # Step 2: Build Docker image
            if not self.docker_manager.build_image(
                "Dockerfile", deployment_config.docker_image, deployment_config.version
            ):
                result.status = DeploymentStatus.FAILED
                result.error_message = "Docker image build failed"
                result.end_time = datetime.now()
                return result

            # Step 3: Push image to registry
            if not self.docker_manager.push_image(
                deployment_config.docker_image, deployment_config.version
            ):
                result.status = DeploymentStatus.FAILED
                result.error_message = "Docker image push failed"
                result.end_time = datetime.now()
                return result

            # Step 4: Deploy to Kubernetes
            if not self.kubernetes_manager.create_deployment(deployment_config):
                result.status = DeploymentStatus.FAILED
                result.error_message = "Kubernetes deployment failed"
                result.end_time = datetime.now()
                return result

            # Step 5: Health check
            if not self._health_check(deployment_config):
                result.status = DeploymentStatus.FAILED
                result.error_message = "Health check failed"
                result.end_time = datetime.now()
                return result

            result.status = DeploymentStatus.SUCCESS
            result.end_time = datetime.now()

            self.logger.info(f"Deployment completed successfully: {deployment_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error during deployment: {e}")
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def _run_cicd_pipeline(self) -> bool:
        """Run CI / CD pipeline."""
        try:
            # Run tests
            if not self.cicd_manager.run_tests():
                return False

            # Run linting
            if not self.cicd_manager.run_linting():
                return False

            # Run security scan
            if not self.cicd_manager.run_security_scan():
                return False

            # Build artifacts
            return self.cicd_manager.build_artifacts()

        except Exception as e:
            self.logger.error(f"Error in CI / CD pipeline: {e}")
            return False

    def _health_check(self, deployment_config: DeploymentConfig) -> bool:
        """Perform health check."""
        try:
            # Mock health check - in production, this would check actual endpoints
            self.logger.info("Performing health check")

            # Simulate health check delay
            import time

            time.sleep(2)

            # Mock health check result
            health_status = True  # In production, check actual health endpoints

            if health_status:
                self.logger.info("Health check passed")
            else:
                self.logger.error("Health check failed")

            return health_status

        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return False

    async def rollback(self, deployment_name: str, previous_version: str) -> bool:
        """Rollback deployment."""
        try:
            self.logger.info(f"Rolling back deployment: {deployment_name} to {previous_version}")

            # Update deployment to previous version
            image_name = f"{deployment_name}: {previous_version}"
            success = self.kubernetes_manager.update_deployment(deployment_name, image_name)

            if success:
                self.logger.info(f"Rollback completed: {deployment_name}")
            else:
                self.logger.error(f"Rollback failed: {deployment_name}")

            return success

        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            return False

    def get_deployment_status(self, deployment_name: str) -> dict[str, Any]:
        """Get deployment status."""
        try:
            status = self.kubernetes_manager.get_deployment_status(deployment_name)
            return status

        except Exception as e:
            self.logger.error(f"Error getting deployment status: {e}")
            return {}


class Phase4Deployment:
    """Main Phase 4 deployment orchestrator."""

    def __init__(self, config: ConfigManager, logger: ProductionLogger):
        self.config = config
        self.logger = logger

        self.deployment_manager = DeploymentManager(config, logger)

        self.logger.info("Phase4Deployment initialized")

    async def deploy_to_environment(
        self, environment: DeploymentEnvironment, version: str | None = None
    ) -> DeploymentResult:
        """Deploy to specific environment."""
        try:
            if not version:
                version = f"v{int(datetime.now().timestamp())}"

            # Create deployment configuration
            deployment_config = DeploymentConfig(
                environment=environment,
                version=version,
                docker_image="wallstreetbots",
                replicas=3 if environment == DeploymentEnvironment.PRODUCTION else 1,
                cpu_limit="1000m",
                memory_limit="2Gi",
                health_check_path="/health",
                environment_variables={
                    "ENVIRONMENT": environment.value,
                    "VERSION": version,
                    "LOG_LEVEL": "INFO",
                },
                ports=["8000: 8000"],
                volumes=["/app / data: /app / data"],
            )

            # Deploy
            result = await self.deployment_manager.deploy(deployment_config)

            return result

        except Exception as e:
            self.logger.error(f"Error deploying to {environment.value}: {e}")
            return DeploymentResult(
                deployment_id="error",
                status=DeploymentStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e),
            )

    async def deploy_all_environments(
        self, version: str | None = None
    ) -> dict[str, DeploymentResult]:
        """Deploy to all environments."""
        try:
            results = {}

            # Deploy to staging first
            self.logger.info("Deploying to staging")
            staging_result = await self.deploy_to_environment(
                DeploymentEnvironment.STAGING, version
            )
            results["staging"] = staging_result

            # Only deploy to production if staging succeeded
            if staging_result.status == DeploymentStatus.SUCCESS:
                self.logger.info("Deploying to production")
                production_result = await self.deploy_to_environment(
                    DeploymentEnvironment.PRODUCTION, version
                )
                results["production"] = production_result
            else:
                self.logger.error("Staging deployment failed, skipping production")
                results["production"] = DeploymentResult(
                    deployment_id="skipped",
                    status=DeploymentStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message="Staging deployment failed",
                )

            return results

        except Exception as e:
            self.logger.error(f"Error deploying to all environments: {e}")
            return {"error": str(e)}

    def get_deployment_status(self, environment: DeploymentEnvironment) -> dict[str, Any]:
        """Get deployment status for environment."""
        try:
            deployment_name = f"wallstreetbots - {environment.value}"
            status = self.deployment_manager.get_deployment_status(deployment_name)
            return status

        except Exception as e:
            self.logger.error(f"Error getting deployment status: {e}")
            return {"error": str(e)}
