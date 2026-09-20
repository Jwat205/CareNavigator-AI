# Production-scale architecture: ALB + auto-scaling ECS Fargate

This is the real answer to "handle 1000+ concurrent users" — not a bigger
single box (we proved in
[`AWS_LOAD_TEST_RESULTS.md`](../../../backend/benchmarks/AWS_LOAD_TEST_RESULTS.md)
that a bigger single box doesn't help, and can hurt), but many small
instances behind a load balancer that scales with demand.

```
Internet -> Application Load Balancer (port 80)
              |
              v
     ECS Fargate Service (2-10 tasks, auto-scaled on request rate)
              |
     each task: 1 vCPU / 4GB, running the FastAPI backend (uvloop, port 8000)
```

Compared to [`../launch_instance.py`](../launch_instance.py) (one EC2 box,
for a quick sanity check), this is a different tier entirely: no single
process ever absorbs the full concurrent load, and the service grows and
shrinks automatically based on real traffic (`ALBRequestCountPerTarget`
target-tracking scaling, 2-10 tasks, target 300 req/min per task).

## Prerequisites

1. Everything from [`../README.md`](../README.md) (the base IAM user +
   `iam-policy.json`).
2. **Also attach** [`iam-policy-production.json`](./iam-policy-production.json)
   to the same IAM user — see the top-level guide's instructions, same
   process, just a second policy. This adds ECR, ECS, load-balancer,
   auto-scaling, and one narrowly-scoped IAM role permission (only for a
   role named exactly `carenav-ecs-task-execution-role` — it cannot touch
   any other role in your account).
3. Docker running locally (this script builds and pushes the image itself).

## Run it

```bash
cd deploy/aws/production
pip install -r ../requirements.txt   # same boto3 dependency
python provision.py                   # ~10-20 min: builds image, pushes to ECR,
                                       # stands up ALB + service, waits for
                                       # tasks to pass health checks
```

Then prove the claim from inside AWS, not from your home connection (see
`AWS_LOAD_TEST_RESULTS.md` for exactly why that distinction matters):

```bash
python run_production_loadtest.py 1 10 50 100 500 1000 1500
```

This launches its own throwaway load-gen instance, runs the sweep against
the ALB's DNS name, prints results, and terminates itself automatically —
you don't need to clean that part up.

## Tear down

```bash
python teardown.py
```

This removes the auto-scaling policy, scales the service to 0, deletes the
service, load balancer, listener, target group, task definition, cluster,
security groups, and log group — everything that bills. It leaves the ECR
repository and the IAM execution role in place by default (both cost
effectively nothing sitting idle, and are handy to reuse next time):

```bash
python teardown.py --delete-ecr-repo --delete-role   # for a truly clean slate
```

**Always run this when you're done.** Unlike the single-instance test, this
provisions multiple always-on resources — the ALB (~$16/month) and the
Fargate tasks (billed per vCPU-second and GB-second while running, roughly
$0.04-0.16/hour per task depending on task size and count) add up if left
running, unlike the single EC2 box which was designed to be torn down within
minutes of each test.

## Why this design, specifically

- **Fargate over raw EC2 + Auto Scaling Group**: no instances to patch or
  manage, and task-level scaling is faster and simpler to reason about than
  EC2 instance-level scaling for a stateless HTTP service like this one.
- **One process per task, not multiple `--workers`**: the single-instance
  testing showed that piling more uvicorn worker processes onto one box
  stopped helping past 4, and even hurt at 16. The cloud-native answer is
  to scale via *task count* behind the load balancer instead of processes
  within one task — each task is small and cheap to replicate.
- **`assignPublicIp: ENABLED` instead of a NAT Gateway**: tasks need
  internet egress (to reach the ECR registry pulling the image, and for the
  app's own NLTK data download on first boot) but a NAT Gateway costs
  ~$32/month just sitting there. Giving tasks a public IP directly, while
  keeping their security group locked to "inbound only from the ALB," gets
  the same practical security posture (nothing can reach a task except
  through the load balancer) without that fixed cost — appropriate for a
  test/demo deployment. A stricter production setup would use private
  subnets + a NAT Gateway or VPC endpoints instead.
- **Model files are baked into the image**, not on a shared volume. Since
  the current app has no models trained yet, and `/upload-and-train` writes
  new models to local disk, multi-task deployments won't share newly
  trained models until the image is rebuilt and redeployed. That's fine for
  serving predictions from models trained ahead of time (rebuild the image
  after training, like any immutable-infrastructure deployment), but if you
  need to train through the live API and have all tasks see the result
  immediately, that requires moving model storage to S3/EFS — a real change,
  intentionally out of scope here since no trained models exist to test
  against yet (see the caveat in `AWS_LOAD_TEST_RESULTS.md` about `/predict`
  never being exercised with a real model).
