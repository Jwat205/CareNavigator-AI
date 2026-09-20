# Load-testing CareNavigator AI on AWS

Three scripts, run in order. Nothing here touches your production AWS setup —
everything is scoped to one throwaway EC2 instance, one security group, and
one key pair, all named `carenav-loadtest*` so they're easy to find and
delete by hand if a script ever fails partway through.

1. `launch_instance.py` — provisions the box (Docker pre-installed, security
   group locked to your current public IP only).
2. `deploy_app.sh` — ships the repo over SSH, builds the image, starts the
   container.
3. `terminate_instance.py` — tears everything down. **Run this when you're
   done**, or the instance keeps billing hourly.

## 1. Set up an IAM user (one-time)

You don't need to hand me your AWS root credentials, and you shouldn't — set
up a dedicated IAM user scoped to exactly what these scripts need.

**Console steps:**

1. AWS Console → **IAM** → **Users** → **Create user**. Name it something
   like `carenav-loadtest`. Do **not** enable console access — this user only
   needs programmatic (API) access.
2. **Attach policies directly** → **Create policy** → **JSON** tab → paste
   the contents of [`iam-policy.json`](./iam-policy.json) → name it
   `CarenavLoadtestPolicy` → create it → attach it to the user you just made.
3. Open the new user → **Security credentials** tab → **Create access key**
   → choose **"Command Line Interface (CLI)"** as the use case → create it.
   Copy the **Access key ID** and **Secret access key** — the secret is only
   shown once.

**What that policy allows, and nothing else:** launching/terminating EC2
instances, creating/deleting the one key pair and one security group these
scripts use, and reading the public "latest Amazon Linux AMI" SSM parameter
so the launch script always picks a current, patched AMI. It cannot touch S3,
IAM, other users' resources, or any other AWS service.

**When you're done testing for good:** delete the access key (Security
credentials tab → Actions → Deactivate/Delete) or delete the whole IAM user.
Either stops that credential from working entirely.

## 2. Give me the credentials

Set them as environment variables in your shell (this keeps them out of any
file I might read or write):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1   # or your preferred region
```

I never need you to paste the secret key into chat — just confirm you've
exported them in the same shell you'll run these scripts from, and I'll run
the scripts from there.

## 3. Run it

```bash
cd deploy/aws
pip install -r requirements.txt
python launch_instance.py                 # ~2-3 min: instance boots, Docker installs
./deploy_app.sh                            # ~10 min: ships code, builds image, starts container
```

`deploy_app.sh` prints the instance's public IP at the end. Run the actual
load test **from your own machine**, not the instance itself — that's the
whole point of moving off the laptop, so the load generator and the server
aren't competing for the same CPU cores:

```bash
cd ../../backend/benchmarks
python benchmark.py --base-url http://<instance-ip>:8000 --concurrent 1 10 50 100 500 --requests 500
```

## 4. Tear down

```bash
cd deploy/aws
python terminate_instance.py
```

Then double check in the AWS Console under **EC2 → Instances** that nothing
named `carenav-loadtest` is still running. This is the step that actually
matters for not getting a surprise bill — the launch script is safe to
re-run any number of times, but a forgotten running instance isn't free.

## Cost

An `m6i.xlarge` (the default) is about $0.19/hr, billed per-second. A full
launch → test → terminate cycle of 20-30 minutes costs roughly $0.06-$0.10.
The 30GB gp3 EBS volume adds a fraction of a cent for that duration. The only
way this gets expensive is forgetting step 4.
