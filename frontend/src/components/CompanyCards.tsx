import React from "react";
import CompanyCard from "./CompanyCard";

const CompanyCards: React.FC = () => {
  const companies = [
    {
      companyName: "TechNova Inc.",
      icon: "/images/technova.png",
      code: "TNV",
      currentStockPrice: 120.45,
      futureStockPrice: 150.78,
      growth: 25.2,
      description:
        "A leading tech company specializing in AI and cloud solutions.",
    },
    {
      companyName: "GreenCore Energy",
      icon: "/images/greencore.png",
      code: "GCE",
      currentStockPrice: 58.23,
      futureStockPrice: 63.5,
      growth: 9.04,
      description:
        "Sustainable energy provider focused on solar and wind power.",
    },
    {
      companyName: "UrbanStyle Ltd.",
      icon: "/images/urbanstyle.png",
      code: "USL",
      currentStockPrice: 34.1,
      futureStockPrice: 30.0,
      growth: -12.02,
      description: "Trendy fashion brand popular among millennials and Gen Z.",
    },
    {
      companyName: "AquaPure Systems",
      icon: "/images/aquapure.png",
      code: "AQP",
      currentStockPrice: 89.99,
      futureStockPrice: 105.0,
      growth: 16.66,
      description: "Water purification tech company with global reach.",
    },
    {
      companyName: "FinGuard Financial",
      icon: "/images/finguard.png",
      code: "FGF",
      currentStockPrice: 210.75,
      futureStockPrice: 190.2,
      growth: -9.74,
      description: "Fintech firm offering smart investment and banking tools.",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {companies.map((company, index) => (
        <CompanyCard key={index} {...company} />
      ))}
    </div>
  );
};

export default CompanyCards;
