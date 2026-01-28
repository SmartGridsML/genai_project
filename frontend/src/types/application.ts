export type AuditClaim = {
  claim: string;
  supported: boolean;
  source?: string;
  confidence?: number; // 0..1
};

export type CvSuggestion = {
  id: string;
  section: string;
  before: string;
  after: string;
};

export type ApplicationResults = {
  cover_letter_text: string;
  audit: AuditClaim[];
  cv_suggestions?: CvSuggestion[];
};
