# Jenkins controller with the tools the pipeline needs: Python 3, Docker CLI, kubectl, kind.
FROM jenkins/jenkins:lts-jdk21
USER root
ARG TARGETARCH
ARG KUBECTL_VERSION=v1.34.1
ARG KIND_VERSION=v0.33.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip docker-cli docker-buildx curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
 && curl -fsSLo /usr/local/bin/kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${TARGETARCH}" \
 && chmod +x /usr/local/bin/kubectl /usr/local/bin/kind
USER jenkins
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
ENV JAVA_OPTS="-Djenkins.install.runSetupWizard=false" \
    CASC_JENKINS_CONFIG=/var/jenkins_home/casc.yaml
COPY casc.yaml /var/jenkins_home/casc.yaml
