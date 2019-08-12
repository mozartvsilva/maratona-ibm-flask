
# Python + Flask Project

A basic Python project using Flask web application framework.

### Requirements

- Python 3.6+
- Docker (To run locally)

### Running Standalone

Before to run, install all python dependencies.

```bash
pip install -r requirements.txt
```

After that, execute `script/run.sh` script.

### Running on Docker

Build docker image execute `script/docker-build.sh` script.

To run the application execute `script/docker-run.sh` script.

To stop the application execute `script/docker-stop.sh` script.

### Running on Redhat OpenShift

Project is ready to run on OpenShift.
Basically, it's only necessary to configure an Environment Variable on Deployment Config pointing to `script/run.sh` script.

### Activate the environment

```bash
. venv/bin/activate
```

```bash
python3 -m venv venv
```
