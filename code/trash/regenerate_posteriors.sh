#!/bin/bash
# Regenerate posterior files with corrected R2 definition

set -e  # Exit on error

echo "🔄 Regenerating posterior files with corrected R2 definition..."
echo ""

# One-off posteriors (marginal P(Rmax | signal))
echo "1️⃣ Generating one-off posteriors (post_mc.npz)..."
python3 precomp.py \
    --seed 123 \
    --rounds 200000 \
    --out ../output/post_mc.npz \
    --procs 8

echo ""
echo "✅ Generated ../output/post_mc.npz"
echo ""

# 2-Stage posteriors (joint P(Rmax | signal, R2))
echo "2️⃣ Generating 2-Stage joint posteriors (post_joint.npz)..."
python3 precomp_joint.py \
    --seed 123 \
    --rounds 500000 \
    --out ../output/post_joint.npz \
    --procs 8

echo ""
echo "✅ Generated ../output/post_joint.npz"
echo ""

echo "🎉 All posterior files regenerated successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Restart the visualizer to load new data"
echo "   2. Verify P(Ace | R2=13) = 1.0 in the 2-Stage view"
