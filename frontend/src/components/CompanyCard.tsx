import React from 'react';
import Image from 'next/image';

interface CompanyCardProps {
  companyName: string;
  icon: string;
  code: string;
  currentStockPrice: number;
  futureStockPrice: number;
  growth: number; // Positive or negative percentage
  description?: string; // Optional description or other details
}

const CompanyCard: React.FC<CompanyCardProps> = ({
  companyName,
  icon,
  code,
  currentStockPrice,
  futureStockPrice,
  growth,
  description,
}) => {
  const isGrowthPositive = growth >= 0;

  return (
    <div
      className="company-card border rounded-lg shadow-md p-4 flex items-center space-x-4"
      style={{ backgroundColor: '#f9f9f9' }}
    >
      <Image src={icon} alt={`${companyName} logo`} width={50} height={50} className="w-12 h-12 rounded-full" />
      <div className="flex-1">
        <h2 className="text-lg font-bold">{companyName}</h2>
        <p className="text-sm text-gray-500">{code}</p>
        <p className="text-sm text-gray-700">{description}</p>
        <div className="mt-2">
          <p>
            Current Price: <span className="font-semibold">${currentStockPrice.toFixed(2)}</span>
          </p>
          <p>
            Future Price: <span className="font-semibold">${futureStockPrice.toFixed(2)}</span>
          </p>
          <p
            className={`font-semibold ${
              isGrowthPositive ? 'text-green-500' : 'text-red-500'
            }`}
          >
            Growth: {isGrowthPositive ? '+' : ''}
            {growth.toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
};

export default CompanyCard;
