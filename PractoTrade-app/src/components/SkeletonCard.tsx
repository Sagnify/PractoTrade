// components/SkeletonCard.tsx
import React from "react";

const SkeletonCard: React.FC = () => {
  return (
    <div className="animate-pulse border rounded-lg shadow-md p-4 flex items-center space-x-4 bg-white">
      <div className="w-12 h-12 bg-gray-300 rounded-full" />
      <div className="flex-1 space-y-2">
        <div className="h-4 bg-gray-300 rounded w-1/3" />
        <div className="h-3 bg-gray-200 rounded w-1/4" />
        <div className="h-3 bg-gray-200 rounded w-2/3" />
        <div className="h-3 bg-gray-200 rounded w-1/2" />
      </div>
    </div>
  );
};

export default SkeletonCard;
