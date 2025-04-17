// components/CompanyCards.tsx
"use client";

import React from "react";
import CompanyCard from "./CompanyCard";
import SkeletonCard from "./SkeletonCard";
import { useStockPredictions } from "@/hooks/useStockPredictions";

const CompanyCards: React.FC = () => {
  const { data: companies, isLoading, isError, error } = useStockPredictions();

  if (isLoading) {
    // Render skeletons while loading
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: 9 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="text-center text-red-500 py-10">
        Error fetching stock data: {(error as Error).message}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 px-20">
      {companies?.map((company, index) => (
        <CompanyCard key={index} {...company} />
      ))}
    </div>
  );
};

export default CompanyCards;
