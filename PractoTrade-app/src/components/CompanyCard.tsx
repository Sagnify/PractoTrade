// components/CompanyCard.tsx
import React from "react";

interface CompanyCardProps {
  companyName: string;
  code: string;
  currentStockPrice: number;
  futureStockPrice: number;
  growth: number;
}

const CompanyCard: React.FC<CompanyCardProps> = ({
  companyName,
  code,
  currentStockPrice,
  futureStockPrice,
  growth,
}) => {
  const isGrowthPositive = growth >= 0;

  return (
    <div className="company-card border rounded-lg shadow-md p-4 text-black flex items-center space-x-4 bg-white">
      <div className="flex-1">
        <h2 className="text-lg font-bold">{companyName}</h2>
        <p className="text-sm text-gray-500">{code}</p>
        <div className="mt-2">
          <p>
            Current Price:{" "}
            <span className="font-semibold">
              ${currentStockPrice.toFixed(2)}
            </span>
          </p>
          <p>
            Future Price:{" "}
            <span className="font-semibold">
              ${futureStockPrice.toFixed(2)}
            </span>
          </p>
          <p
            className={`font-semibold ${
              isGrowthPositive ? "text-green-500" : "text-red-500"
            }`}
          >
            Growth: {isGrowthPositive ? "+" : ""}
            {growth.toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
};

export default CompanyCard;
