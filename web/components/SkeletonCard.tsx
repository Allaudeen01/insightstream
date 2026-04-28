import React from 'react';

const SkeletonCard = () => (
  <div className="
    rounded-xl border border-white/10
    bg-white/5 p-6 animate-pulse
  ">
    <div className="h-4 bg-white/10 rounded w-1/3 mb-4" />
    <div className="h-8 bg-white/10 rounded w-1/2 mb-2" />
    <div className="h-3 bg-white/10 rounded w-2/3" />
  </div>
);

export default SkeletonCard;
