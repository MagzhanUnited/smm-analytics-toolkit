// CI/CD pipeline for smm-analytics-toolkit.
// CI on every commit of every branch: lint -> unit tests (JUnit + coverage gate) -> Docker image -> smoke test.
// CD on main only: push to the local registry and roll out to the Kubernetes test cluster (kind).
pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 20, unit: 'MINUTES')
    }

    environment {
        IMAGE      = 'smmkit'
        REGISTRY   = 'localhost:5001'
        KIND_NAME  = 'smm-ci'
        K8S_NS     = 'smmkit'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_SHORT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.TAG = "${env.BUILD_NUMBER}-${env.GIT_SHORT}"
                }
                sh 'git log -1 --pretty="format:%h %an: %s"'
            }
        }

        stage('Install') {
            steps {
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install --quiet -e ".[dev,plot]"
                    .venv/bin/python --version
                    .venv/bin/pip list 2>/dev/null | grep -E "pytest|ruff|matplotlib|smm"
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '.venv/bin/ruff check . && .venv/bin/ruff format --check .'
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    mkdir -p reports
                    .venv/bin/pytest --junitxml=reports/junit.xml \
                           --cov=smmkit --cov-report=term --cov-report=xml:reports/coverage.xml \
                           --cov-fail-under=90
                '''
            }
            post {
                always {
                    junit testResults: 'reports/junit.xml', skipPublishingChecks: true
                    recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']],
                                   sourceDirectories: [[path: 'src']],
                                   enabledForFailure: true, skipPublishingChecks: true)
                }
            }
        }

        stage('Build Docker image') {
            steps {
                sh '''
                    docker build --pull \
                        --build-arg VCS_REF=${GIT_SHORT} \
                        -t ${IMAGE}:${TAG} -t ${IMAGE}:latest .
                    docker image ls ${IMAGE}:${TAG}
                '''
            }
        }

        stage('Smoke test image') {
            steps {
                sh '''
                    name=smoke-${BUILD_TAG}
                    docker run -d --rm --name "$name" ${IMAGE}:${TAG}
                    trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
                    for i in $(seq 1 20); do
                        if docker exec "$name" python -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:8080/healthz').read().decode())" 2>/dev/null; then
                            echo; echo "container healthy after $i attempt(s)"; exit 0
                        fi
                        sleep 1
                    done
                    docker logs "$name"; exit 1
                '''
            }
        }

        stage('Push image') {
            when { branch 'main' }
            steps {
                sh '''
                    docker tag ${IMAGE}:${TAG} ${REGISTRY}/${IMAGE}:${TAG}
                    docker tag ${IMAGE}:${TAG} ${REGISTRY}/${IMAGE}:latest
                    docker push ${REGISTRY}/${IMAGE}:${TAG}
                    docker push ${REGISTRY}/${IMAGE}:latest
                '''
            }
        }

        stage('Deploy to Kubernetes (kind)') {
            when { branch 'main' }
            steps {
                sh '''
                    kind load docker-image ${IMAGE}:${TAG} --name ${KIND_NAME}
                    kubectl apply -f deploy/k8s/
                    kubectl -n ${K8S_NS} set image deployment/smmkit smmkit=${IMAGE}:${TAG}
                    kubectl -n ${K8S_NS} annotate deployment/smmkit kubernetes.io/change-cause="build ${TAG}" --overwrite
                    kubectl -n ${K8S_NS} rollout status deployment/smmkit --timeout=120s
                    kubectl -n ${K8S_NS} get deploy,pods,svc -o wide
                '''
            }
        }

        stage('Verify deployment') {
            when { branch 'main' }
            steps {
                sh '''
                    kubectl -n ${K8S_NS} exec deploy/smmkit -- python -c "import urllib.request as u; print(u.urlopen('http://smmkit.smmkit.svc:8080/healthz').read().decode())"
                    kubectl -n ${K8S_NS} rollout history deployment/smmkit | tail -3
                '''
            }
        }
    }

    post {
        success { echo "Image ${IMAGE}:${TAG} built${env.BRANCH_NAME == 'main' ? ' and deployed' : ''}." }
        always  { sh 'docker image prune -f --filter "label=org.opencontainers.image.title=smm-analytics-toolkit" --filter "until=24h" || true' }
    }
}
