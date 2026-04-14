

# Camera

Installed based on systemd, see `install.sh`.

In general commands are:

```sh
systemctl daemon-reload
systemctl status streamer
systemctl enable streamer
systemctl start streamer
systemctl stop streamer
journald -b -u streamer
```

To configure it, override the environment variables by created a systemd config extension file `/etc/systemd/system/streamer.service.d/override.conf` like:

```ini
[Service]
Environment="HOST=someotherhost"
Environment="PATH=someotherpath"
```

Note: if `/etc/systemd/system/streamer.service.d/` exists and is empty then the service will be disabled!

# Server
