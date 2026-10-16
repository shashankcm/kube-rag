"""
Builds evals/golden_v2.json: a retrieval golden set for kube-rag.

Every answerable question carries one or more evidence spans. Each span must
appear verbatim (whitespace-normalised) in the parsed text of its source file
under processed_data/true, so a gold label can never claim something the
source does not say. Unanswerable questions have no support in the corpus.

Run:  python -m evals.build_golden_v2
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = ROOT / "processed_data" / "true"
OUT_PATH = ROOT / "evals" / "golden_v2.json"

SRC = {
    "pwq": "parallel_work_queue.txt",
    "jm": "job_management.html",  # Databricks docs, not Kubernetes
    "mj": "monitor_job.docx",
    "arch": "architecture.pptx",
    "pa": "pods_autoscale.html",
    "cj": "cronjobs.docx",
}


def Q(src, question, answer, spans, kind="factual"):
    spans = [spans] if isinstance(spans, str) else spans
    return {
        "question": question,
        "reference": answer,
        "type": kind,
        "evidence": [{"source": SRC[src], "span": s} for s in spans],
    }


def M(question, answer, pairs):
    return {
        "question": question,
        "reference": answer,
        "type": "multi_source",
        "evidence": [{"source": SRC[s], "span": sp} for s, sp in pairs],
    }


ANSWERABLE = [
    # --- parallel_work_queue.txt
    Q("pwq", "What storage service does the work queue example use to hold the queue?", "Redis.", "Start a storage service to hold the work queue. (We will use Redis)."),
    Q("pwq", "What is the name of the queue that is filled with tasks in the work queue example?", "The queue is named job2.", 'fill the queue (named "job2")'),
    Q("pwq", "Which fruits are pushed onto the job2 queue?", "apple, banana, cherry, date, fig, grape, lemon, melon and orange.", ['rpush job2 "apple"', 'rpush job2 "orange"'], "enumeration"),
    Q("pwq", "How do workers signal that the queue is empty?", "They exit with success.", "Workers signal the queue is empty by exiting with success."),
    Q("pwq", "What parallelism does the work queue Job manifest set?", "parallelism: 2.", "parallelism: 2"),
    Q("pwq", "Which command waits up to 300 seconds for the work queue Job to complete?", "kubectl wait --for=condition=complete --timeout=300s job/job-wq-2", "kubectl wait --for=condition=complete --timeout=300s job/job-wq-2", "procedural"),
    Q("pwq", "What does the work queue guide suggest for a continuous stream of background work?", "Run the background workers with a ReplicaSet instead.", "running your background workers with a ReplicaSet instead."),
    Q("pwq", "How long is the lease the worker takes on a queue item?", "10 seconds (lease_secs=10).", "q.lease(lease_secs=10, block=True, timeout=2)"),
    # --- job_management.html (Databricks)
    Q("jm", "Which three tools does the Databricks guide compare for managing jobs programmatically?", "The Databricks CLI, the Databricks SDKs and the REST API.", "The following table compares the Databricks CLI, the Databricks SDKs, and the REST API"),
    Q("jm", "Which languages are the Databricks SDKs available for?", "Python, Java, Go and R.", "available for Python, Java, Go, or R"),
    Q("jm", "Which REST API request does the command databricks jobs get map to?", "GET /api/2.2/jobs/get.", "maps to GET/api/2.2/jobs/get"),
    Q("jm", "How many ways are there to run a Databricks job from the command line?", "Three: on a schedule, run-now, and submit.", "There are three ways to run your job when using the command line."),
    Q("jm", "What happens to a Databricks job started with the submit command?", "It is not saved, is not visible in the UI, and runs once.", "the job is not saved, and isn't visible in the UI"),
    Q("jm", "Which CLI command deletes a Databricks job?", "databricks jobs delete <job-id>", "databricks jobs delete <job-id>", "procedural"),
    Q("jm", "What does Databricks recommend for managing jobs as part of CI/CD?", "Databricks Asset Bundles or the Databricks Terraform provider.", "To programmatically manage jobs as part of CI/CD use"),
    Q("jm", "Which environment variables does the Databricks REST API example assume are set?", "DATABRICKS_HOST and DATABRICKS_TOKEN.", ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]),
    # --- monitor_job.docx
    Q("mj", "Which tool does the job monitoring tutorial use to run a local Kubernetes cluster?", "Minikube, a single-node cluster on your local machine.", "We'll use Minikube, a tool that creates a single-node Kubernetes cluster on your local machine."),
    Q("mj", "What does a backoffLimit of 4 mean for hello-job?", "The job retries up to 4 times if it fails.", "Has a backoffLimit of 4, meaning it will retry up to 4 times if it fails"),
    Q("mj", "What does the COMPLETIONS column show in kubectl get jobs?", "The number of completed pods out of the desired completions.", "COMPLETIONS: The number of completed pods / desired completions"),
    Q("mj", "Which statuses can a Kubernetes job have according to the tutorial?", "Active, Completed and Failed.", "A job can have several statuses:"),
    Q("mj", "How do you follow a job pod's logs in real time?", "kubectl logs -f <pod-name>", "kubectl logs -f <pod-name>", "procedural"),
    Q("mj", "How do you list only the pods that belong to the job hello-job?", "kubectl get pods -l job-name=hello-job", "kubectl get pods -l job-name=hello-job", "procedural"),
    Q("mj", "How do you read the number of succeeded pods for labeled-job with JSONPath?", "kubectl get job labeled-job -o jsonpath='{.status.succeeded}'", "kubectl get job labeled-job -o jsonpath='{.status.succeeded}'", "procedural"),
    Q("mj", "How do you filter jobs by both the department and environment labels?", "kubectl get jobs -l department=engineering,environment=training", "kubectl get jobs -l department=engineering,environment=training", "procedural"),
    Q("mj", "What schedule does simple-cronjob use?", "Every minute (*/1 * * * *).", 'schedule: "*/1 * * * *"'),
    Q("mj", "Can you read a job's logs after the job has completed?", "Yes, logs remain available after completion.", "you can view the logs even after the job has completed"),
    # --- architecture.pptx
    Q("arch", "Which Kubernetes version does the architecture overview deck cover?", "Kubernetes v1.8.", "Kubernetes v1.8"),
    Q("arch", "How does the architecture deck describe Kubernetes?", "As the linux kernel of distributed systems.", "Kubernetes is the linux kernel of distributed systems."),
    Q("arch", "Which Google systems informed the design of Kubernetes?", "Borg and Omega.", "lessons learned from developing and managing Borg and Omega"),
    Q("arch", "Which master components does the deck list?", "kube-apiserver, etcd, kube-controller-manager, cloud-controller-manager and kube-scheduler.", "Kube-apiserver Etcd Kube-controller-manager Cloud-controller-manager Kube-scheduler", "enumeration"),
    Q("arch", "What role does etcd play in the cluster?", "It is the cluster datastore: a consistent, highly available key-value store for cluster state.", "Etcd acts as the cluster datastore; providing a strong, consistent and highly available key-value store used for persisting cluster state."),
    Q("arch", "What does kube-proxy do?", "It manages network rules on each node and forwards connections or load balances for services.", "Manages the network rules on each node and performs connection forwarding or load balancing for Kubernetes cluster services."),
    Q("arch", "What do the Kubernetes networking fundamental rules say about NAT?", "Pods can talk to all other pods, and nodes to all pods, without NAT.", "All Pods can communicate with all other Pods without NAT"),
    Q("arch", "How many major Service types are there?", "Four: ClusterIP (default), NodePort, LoadBalancer and ExternalName.", "Four major Service Types:", "enumeration"),
    Q("arch", "Which Service type exposes a service only on a cluster-internal IP?", "ClusterIP, which is the default.", "CluterIP - Exposes service on a strictly cluster-internal IP (default)"),
    Q("arch", "What is a DaemonSet used for?", "To ensure matching nodes each run an instance of a Pod, for example log forwarding or health monitoring.", "Ensures that all nodes matching certain criteria will run an instance of a supplied Pod."),
    Q("arch", "How are StatefulSet pods named?", "<statefulset name>-<ordinal index>.", "Pods given unique ordinal names using the pattern <statefulset name>-<ordinal index>."),
    Q("arch", "Which Job fields control the number of pod executions and parallelism?", "spec.completions and spec.parallelism.", "Number of pod executions can be controlled via spec.completions"),
    Q("arch", "Are Jobs and their Pods cleaned up automatically after completion?", "No, they are not automatically cleaned up.", "Jobs and Pods are NOT automatically cleaned up after a job has completed."),
    Q("arch", "How do Secrets differ from ConfigMaps?", "Secrets are stored base64-encoded and encrypted at rest if configured.", "Functionally identical to ConfigMaps, but stored encoded as base64, and encrypted at rest (if configured)."),
    Q("arch", "What does the pause container do for a Pod?", "It acts as the parent container for the Pod.", "The pause container acts as the parent container for the Pod."),
    # --- pods_autoscale.html
    Q("pa", "What must be installed before testing HPA and VPA?", "The Metrics Server.", "Metrics Server installed in your Kubernetes cluster"),
    Q("pa", "What does the Horizontal Pod Autoscaler do?", "It scales the number of pods based on observed CPU utilization or other metrics.", "automatically scales the number of pods in a replication controller, deployment, or replica set based on observed CPU utilization"),
    Q("pa", "What replica range and CPU target does the example HPA use?", "1 to 10 replicas, targeting about 50% CPU.", "between 1 and 10 based on the CPU utilization, aiming to keep it around 50%"),
    Q("pa", "Which command generates load to test the HPA?", "A busybox load-generator pod started with kubectl run that requests the nginx service in a loop.", "kubectl run -i --tty load-generator --rm --image=busybox:1.28 --restart=Never", "procedural"),
    Q("pa", "What does the Vertical Pod Autoscaler adjust?", "The CPU and memory requests and limits of pods.", "adjusts the CPU and memory resource requests and limits for your pods"),
    Q("pa", "Which script installs the Vertical Pod Autoscaler?", "./hack/vpa-up.sh in the autoscaler repository.", "./hack/vpa-up.sh", "procedural"),
    Q("pa", "Which updateMode options does VPA support?", "Off, Initial, Recreate and Auto.", "There are multiple valid options for updateMode in VPA.", "enumeration"),
    Q("pa", "What does VPA updateMode Off do?", "It only provides recommendations and does not change resources.", "VPA will only provide the recommendations, and it will not automatically change resource requirements."),
    Q("pa", "What minAllowed and maxAllowed CPU does the example VPA set?", "minAllowed 100m and maxAllowed 200m CPU.", "minAllowed: cpu: 100m memory: 50Mi maxAllowed: cpu: 200m"),
    Q("pa", "Which repository is cloned to install VPA?", "https://github.com/kubernetes/autoscaler.git", "git clone https://github.com/kubernetes/autoscaler.git", "procedural"),
    # --- cronjobs.docx
    Q("cj", "Which restart policy is not allowed for Jobs and why?", "Always, because it would prevent the Pod from ever completing.", "Always: Not allowed (would prevent Pod from ever completing)."),
    Q("cj", "Can a Job's Pod template be changed after the Job is created?", "No, it is immutable.", "Pod template cannot be modified after Job creation."),
    Q("cj", "What does completions: 5 mean for a Job?", "The Job completes after 5 successful Pod completions.", "Job completes after 5 successful Pod completions"),
    Q("cj", "What does Indexed completion mode give each Pod?", "A unique completion index, exposed as JOB_COMPLETION_INDEX.", "Each Pod gets a unique completion index:"),
    Q("cj", "What is a CronJob?", "A thin wrapper around Jobs that enables scheduling.", "A CronJob is a thin wrapper around Jobs that enables scheduling."),
    Q("cj", "Which cron expression runs a job daily at 2:30 AM?", "30 2 * * *", 'schedule: "30 2 * * *"', "procedural"),
    Q("cj", "Which CronJob field sets the time zone?", "timeZone, for example America/New_York.", 'timeZone: "America/New_York"'),
    Q("cj", "Which concurrencyPolicy does the database backup CronJob use?", "Forbid.", "concurrencyPolicy: Forbid"),
    Q("cj", "Why is the cron schedule '60 * * * *' invalid?", "The minute field must be 0-59.", 'schedule: "60 * * * *" # Invalid - minute must be 0-59'),
    Q("cj", "Which Prometheus alert fires when a CronJob has not run in over an hour?", "CronJobMissing.", "alert: CronJobMissing"),
    Q("cj", "Which security context setting stops a job container gaining extra privileges?", "allowPrivilegeEscalation: false.", "allowPrivilegeEscalation: false"),
    # --- multi-source: the answer appears in more than one document
    M("What is a Kubernetes Job?", "A controller or resource that runs one or more pods to completion.", [
        ("arch", "The job controller ensures one or more pods are executed and successfully terminates."),
        ("cj", "A Kubernetes Job is a resource designed to run finite workloads that execute a task to completion."),
        ("mj", "A Kubernetes Job is a controller that creates one or more pods and ensures they successfully terminate."),
    ]),
    M("Which field limits how many times a failing Job is retried?", "backoffLimit.", [
        ("mj", "Has a backoffLimit of 4, meaning it will retry up to 4 times if it fails"),
        ("cj", "backoffLimit: 3"),
    ]),
    M("How can a Job run its pods in parallel?", "Set spec.parallelism (and completions).", [
        ("arch", "Jobs can be parallelized using spec.parallelism"),
        ("cj", "Controls maximum concurrent Pods:"),
        ("pwq", "parallelism: 2"),
    ]),
]

# Plausible in-domain questions the six documents do not answer.
UNANSWERABLE = [
    "How do I configure Istio mutual TLS between two services?",
    "How do I upgrade a cluster from Kubernetes 1.8 to 1.9 with kubeadm?",
    "What are the pricing tiers for Databricks jobs compute?",
    "How do I back up etcd with etcdctl snapshot save?",
    "How do I configure a PodDisruptionBudget for a Deployment?",
    "How do I write a Kubernetes operator in Go?",
    "How do I set up cert-manager to issue TLS certificates for an Ingress?",
    "How do I enable Pod Security Admission in restricted mode?",
    "Which Kubernetes version removed Docker as a container runtime?",
    "How do I configure a NetworkPolicy to block all ingress traffic by default?",
]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_corpus() -> dict[str, str]:
    corpus = {}
    for name in SRC.values():
        data = json.loads((PARSED_DIR / f"{name}.json").read_text())
        corpus[name] = normalise(" ".join(data["chunks"]))
    return corpus


def unverified_spans(questions: list[dict], corpus: dict[str, str]) -> list[tuple]:
    return [
        (q["id"], e["source"], e["span"])
        for q in questions
        for e in q["evidence"]
        if normalise(e["span"]) not in corpus[e["source"]]
    ]


def build() -> dict:
    questions = [{"id": f"Q{i:03d}", **q} for i, q in enumerate(ANSWERABLE, 1)]
    unanswerable = [
        {"id": f"U{i:02d}", "question": q, "type": "unanswerable", "evidence": []}
        for i, q in enumerate(UNANSWERABLE, 1)
    ]
    return {
        "version": 2,
        "description": (
            "Retrieval golden set for kube-rag. Gold = the source file(s) holding the "
            "evidence span; every span is verified verbatim against processed_data/true. "
            "'unanswerable' questions have no support in the corpus."
        ),
        "questions": questions,
        "unanswerable": unanswerable,
    }


if __name__ == "__main__":
    dataset = build()
    missing = unverified_spans(dataset["questions"], load_corpus())
    print(f"answerable: {len(dataset['questions'])} | unanswerable: {len(dataset['unanswerable'])} | unverified spans: {len(missing)}")
    for m in missing:
        print("  MISSING", m)
    print("by first source:", dict(Counter(q["evidence"][0]["source"] for q in dataset["questions"])))
    if missing:
        raise SystemExit("Fix the unverified spans before writing the dataset.")
    OUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    print("wrote", OUT_PATH.relative_to(ROOT))
