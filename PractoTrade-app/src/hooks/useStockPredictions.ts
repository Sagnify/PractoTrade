// hooks/useStockPredictions.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { useCompanyList } from "./useCompanyList";

interface StockApiResponse {
  company: string;
  predicted_Close: number;
  prediction_time: string;
  predicted_percentage_change: number;
  direction: string;
}

interface CompanyCardData {
  companyName: string;
  code: string;
  currentStockPrice: number;
  futureStockPrice: number;
  growth: number;
}

export const useStockPredictions = () => {
  const { data: companies } = useCompanyList();

  return useQuery<CompanyCardData[]>({
    queryKey: ["stock-predictions"],
    enabled: !!companies, // only run if companies are loaded
    queryFn: async () => {
      const codes = Object.keys(companies!);

      const results = await Promise.all(
        codes.map(async (code) => {
          const res = await axios.get<StockApiResponse>(
            `https://implicit-electra-sagnify-8514ada8.koyeb.app/get_predicted_stock_price/${code}/`
          );

          const data = res.data;

          // Calculate current stock price based on the predicted percentage change
          const currentPrice = Number(
            (
              data.predicted_Close /
              (1 + data.predicted_percentage_change / 100)
            ).toFixed(2)
          );

          return {
            companyName: companies![code],
            code: code,
            currentStockPrice: currentPrice,
            futureStockPrice: data.predicted_Close,
            growth: data.predicted_percentage_change,
          };
        })
      );

      return results;
    },
    staleTime: 1000 * 60 * 5, // cache for 5 minutes
    // cacheTime: 1000 * 60 * 10, // cache for 10 minutes
    refetchOnWindowFocus: false,
    retry: 1,
  });
};
