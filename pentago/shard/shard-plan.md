# Shard generation plan

Generate ~100 GB of shard files (8500 of 1048576 total shards, slices 0–18) for initial
ML training.

## Prep

```bash
# Create the output bucket
gcloud storage buckets create gs://pentago-shards --location=us-central1

# Move slices 17–18 out of coldline to avoid retrieval fees on future reads.
# Each command does a server-side rewrite that requires multiple round-trips,
# so run from Cloud Shell or a GCE instance to avoid losing progress if your
# local machine disconnects. Safe to re-run if interrupted (original object
# stays intact until the rewrite completes).
gcloud storage objects update gs://pentago-us-central1/slice-17.pentago --storage-class=standard
gcloud storage objects update gs://pentago-us-central1/slice-18.pentago --storage-class=standard
```

The rewrite takes a while (~3.4 TB to read and rewrite). Can run overnight.

## Instance setup

```bash
# 192 vCPU for fast lzma decompression, 12 TB Titanium SSD (4 × 3 TB NVMe,
# built into machine type — no --local-ssd flags needed).
# Auto-terminates after 2 hours as a safety net against forgotten instances.
gcloud compute instances create pentago-sharder \
  --zone=us-central1-a \
  --machine-type=c4d-standard-192-lssd \
  --boot-disk-size=50GB \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --max-run-duration=2h \
  --instance-termination-action=DELETE \
  --scopes=storage-full

gcloud compute ssh pentago-sharder --zone=us-central1-a
```

```bash
# RAID0 the Titanium SSD devices into a single volume.
# Detect them by their 3T size (avoids hardcoding device names).
sudo apt-get update && sudo apt-get install -y mdadm git build-essential
sudo curl -Lo /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazelisk
sudo ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel
DEVICES=$(lsblk -dno NAME,SIZE | awk '$2=="375G"{print "/dev/"$1}')
echo "Found $(echo $DEVICES | wc -w) SSDs: $DEVICES"  # should be 8
N=$(echo $DEVICES | wc -w)
sudo mdadm --create /dev/md0 --level=0 --raid-devices=$N $DEVICES
sudo mkfs.ext4 -F /dev/md0
sudo mkdir /mnt/data
sudo mount /dev/md0 /mnt/data
sudo chown $USER /mnt/data
mkdir /mnt/data/input /mnt/data/output
```

## Copy input data (~10 min)

```bash
# Copy all 19 slice files (~4.3 TB, same-region, auto-parallelized)
gcloud storage cp 'gs://pentago-us-central1/slice-*.pentago' /mnt/data/input/

# Verify all 19 slices arrived
ls /mnt/data/input/slice-*.pentago | wc -l  # should be 19
du -sh /mnt/data/input/  # should be ~4.3 TB
```

## Build and run (~15 min)

```bash
cd ~
git clone https://github.com/girving/pentago.git
cd pentago
bin/bazel build -c opt //pentago/shard:sharder

bazel-bin/pentago/shard/sharder \
  --max-slice 18 \
  --shards 1048576 \
  --range :8500 \
  /mnt/data/input \
  /mnt/data/output
```

With 192 vCPU doing parallel lzma decompression and high-bandwidth Titanium SSD,
the sharder should process all 4.3 TB in ~10 minutes.
Progress is logged per-block per-slice.

## Verify output

```bash
# Check shard count and total size
ls /mnt/data/output/*.pentago.shard | wc -l  # should be 8500
du -sh /mnt/data/output/  # should be ~100 GB

# Spot-check a few shard sizes (should be ~12.5 MB each)
ls -lh /mnt/data/output/ | head -5
```

## Upload and tear down

```bash
gcloud storage cp '/mnt/data/output/*.pentago.shard' gs://pentago-shards/
exit
```

```bash
gcloud compute instances delete pentago-sharder --zone=us-central1-a
```

Local SSDs are automatically destroyed with the instance. The instance also
auto-terminates after 2 hours (from `--max-run-duration`) as a safety net.

## Sizing notes

| Parameter | Value |
|-----------|-------|
| Total shards | 1,048,576 |
| Shard range | :8500 (first 8500 shards) |
| Output size | ~100 GB |
| Per-shard size | ~12 MB |
| Filename pattern | `shard-0000000-of-1048575.pentago.shard` |

## Revert slices 17–18 to coldline

Once all shard generation is complete (including future full runs), revert to coldline.
No retrieval fee in this direction.

```bash
gcloud storage objects update gs://pentago-us-central1/slice-17.pentago --storage-class=coldline
gcloud storage objects update gs://pentago-us-central1/slice-18.pentago --storage-class=coldline
```

## Future: full shard generation

To generate all 1M shards later, use `--range 8500:` on a similar instance.
The sharder supports split ranges, so previously generated shards don't need
to be regenerated. Total output: ~12.5 TB across 1,048,576 files.
