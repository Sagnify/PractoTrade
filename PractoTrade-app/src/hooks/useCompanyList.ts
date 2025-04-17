// hooks/useCompanyList.ts
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

interface CompanyMapResponse {
  companies: Record<string, string>; // { "META": "Meta", ... }
}

export const useCompanyList = () =>
  useQuery<Record<string, string>>({
    queryKey: ["company-list"],
    queryFn: async () => {
      const res = await axios.get<CompanyMapResponse>(
        "https://implicit-electra-sagnify-8514ada8.koyeb.app/api/companies/"
      );
      return res.data.companies;
    },
    staleTime: 1000 * 60 * 60, // 1 hour
    // cacheTime: 1000 * 60 * 60,
    refetchOnWindowFocus: false,
  });
