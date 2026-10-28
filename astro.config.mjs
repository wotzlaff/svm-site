import { defineConfig } from "astro/config";

// Deployed to https://wotzlaff.github.io/svm-site/ (adapt `site`/`base` for a custom domain)
export default defineConfig({
  site: "https://wotzlaff.github.io",
  base: "/svm-site",
  trailingSlash: "ignore",
});
