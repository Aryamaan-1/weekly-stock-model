# Weekly Stock Model Deployment

This repository contains all the code and configuration needed to deploy the weekly stock model on an EC2 instance using **systemd**. Upon first boot, a helper script will install dependencies, run the model, and shut down the instance. On subsequent reboots, the model will execute automatically via a persistent systemd service.

---

### 📁 Repository Structure

```
model/
├── model_script.py       # Main model training script
├── requirements.txt      # Python dependencies

deploy/
└── systemd/
    └── weekly-stock-model.service  # systemd unit file template

run-on-boot.sh           # Helper script to bootstrap first run
.gitignore
```

---

### 🛠️ Prerequisites

* **EC2 Instance** running Amazon Linux 2 or any system with `systemd` and Python 3.
* **IAM Role** attached to the instance with permissions to:

  * Read from your S3 bucket (if fetching code from S3)
  * Write to CloudWatch Logs (optional)
  * Shutdown the instance
* **Git** installed on the instance

---

### 🔧 Deployment Steps

1. **Clone the repository**

   ```bash
   cd /home/ec2-user
   git clone <REPO_URL> weekly-stock-model
   cd weekly-stock-model
   ```

2. **Configure environment variables**

   Edit `deploy/systemd/weekly-stock-model.service` and replace placeholders with actual values:

   ```ini
   [Unit]
   Description=Weekly Stock Model Runner
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=oneshot
   ExecStart=/home/ec2-user/weekly-stock-model/run-on-boot.sh
   Environment=BACKUP_BUCKET=<your-s3-bucket>
   Environment=BACKUP_PREFIX=<your-s3-prefix>
   Environment=PROJECT_DIR=/home/ec2-user/weekly-stock-model/model
   StandardOutput=journal
   StandardError=journal
   RemainAfterExit=no

   [Install]
   WantedBy=multi-user.target
   ```

3. **Install and enable the service** (first run)

   ```bash
   # Copy service file to systemd directory
   sudo cp deploy/systemd/weekly-stock-model.service /etc/systemd/system/

   # Ensure run-on-boot script is executable
   chmod +x run-on-boot.sh

   # Reload systemd and enable the service
   sudo systemctl daemon-reload
   sudo systemctl enable weekly-stock-model.service

   # Start the service immediately (bootstrap)
   sudo systemctl start weekly-stock-model.service
   ```

   The helper script `run-on-boot.sh` will:

   * Create and activate a Python virtual environment
   * Install dependencies from `model/requirements.txt`
   * Execute `model/model_script.py`
   * Shut down the instance when complete

4. **Verify execution**

   * **Console Logs**: Check `journalctl -u weekly-stock-model.service` for output.
   * **Instance State**: The EC2 instance should stop itself once the script finishes.

5. **Automatic runs on reboot**

   On every subsequent start/reboot, `weekly-stock-model.service` will run automatically without manual intervention. The instance will shut itself down on completion.

---

### 🚀 Usage Notes

* If you need to update the code, simply push changes to your Git repo and restart the instance.
* To view logs in CloudWatch, install and configure the CloudWatch Logs agent in `run-on-boot.sh` or manually.

---

### 📝 .gitignore

```
.env
venv/
*.pyc
__pycache__/
```

---
