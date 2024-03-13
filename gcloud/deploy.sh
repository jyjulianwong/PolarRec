#!/bin/bash
# Set the working directory.
cd /var/www/github_jyjulianwong_polarrec

# Update the local copy of the Cloud Source Repository.
git config --global --add safe.directory /var/www/github_jyjulianwong_polarrec
git checkout main
git fetch --all
git reset --hard origin/main

# Install Python dependencies within the virtual environment.
source /var/www/github_jyjulianwong_polarrec/venv/bin/activate
pip install -r requirements.txt

sudo systemctl start PolarRec.service
echo "Started PolarRec.service in background process with systemctl."

sudo systemctl enable PolarRec.service
echo "Enabled systemctl PolarRec.service so that background process restarts upon system restart."