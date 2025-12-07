# IRIS

A research project done with the AI Team by Myriam Benlamri (Data Science MSc 2nd year) and Marcus Hamelink (Computer Science BSc 3rd year) as a collaborative research semester project.

More info at [https://epflaiteam.ch/projects/iris](https://epflaiteam.ch/projects/iris)


## Set up

### Client

On Your Raspberry Pi, generate the self-signed certificate:
```bash
mkdir -p ~/iris-certs
cd ~/iris-certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem \
  -out cert.pem \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=$(hostname -I | awk '{print $1}')"
```

To use HTTPS, make sure to specify `use_ssl=true` under `client` in `config.yaml`. In that case, make sure to connect to the HTTPS IP.

Run `uv run iris-client` to start a client instance.


### Server

Run `uv run iris-server` to start a server instance.


## Workflow with Izar

This supposee

### On Izar
```
cd /path/to/IRIS
Sinteract -t 00:20:00 -g gpu:1 -m 32G -q team-ai
hostname
./run_iris.sh
```

### On Personal machine

**Terminal 1**
```
uv run iris-client
```

**Terminal 2**
```
ssh -N -L 8005:[RUN hostname ON NODE TO SEE]:8001 EPFL-USERNAME@izar.hpc.epfl.ch
```

Then go to http://localhost:8006

Important, make sure to modify the hostname
