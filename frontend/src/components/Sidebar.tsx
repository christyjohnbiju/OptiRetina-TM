'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Home, Upload, FileText, History, LogOut, LayoutDashboard } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { signOut } from "next-auth/react";

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="h-screen w-64 bg-white border-r border-slate-200 flex flex-col p-4 fixed left-0 top-0 z-20">
       <div className="mb-10 px-2 mt-4 flex items-center space-x-3">
         <div className="h-9 w-9 bg-blue-600 rounded-xl flex items-center justify-center shadow-md shadow-blue-200">
            <LayoutDashboard className="h-5 w-5 text-white" />
         </div>
         <span className="font-bold text-xl text-slate-800 tracking-tight">OptiRetina</span>
       </div>
       
       <nav className="flex-1 space-y-1">
          <p className="px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Menu</p>
          <NavItem href="/dashboard" icon={<Home size={18} />} label="Overview" active={pathname === '/dashboard'} />
          <NavItem href="/dashboard/upload" icon={<Upload size={18} />} label="New Analysis" active={pathname === '/dashboard/upload'} />
          <NavItem href="/dashboard/history" icon={<History size={18} />} label="Patient History" active={pathname?.startsWith('/dashboard/history')} />
       </nav>
       
       <div className="mt-auto pt-6 border-t border-slate-100">
          <div className="flex items-center gap-3 px-3 mb-4">
            <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-bold text-xs">
                Dr
            </div>
            <div className="text-xs">
                <p className="font-medium text-slate-700">Dr. Admin</p>
                <p className="text-slate-400">Ophthalmologist</p>
            </div>
          </div>
          <Button 
            variant="ghost" 
            className="w-full justify-start text-red-500 hover:text-red-600 hover:bg-red-50 font-medium"
            onClick={() => signOut({ callbackUrl: '/login' })}
          >
            <LogOut className="mr-2 h-4 w-4" /> Sign Out
          </Button>
       </div>
    </div>
  )
}

function NavItem({ href, icon, label, active }: { href: string, icon: React.ReactNode, label: string, active?: boolean }) {
    return (
        <Link href={href} className={cn(
            "flex items-center space-x-3 px-4 py-2.5 rounded-lg transition-all duration-200 text-sm font-medium",
            active 
              ? "bg-blue-50 text-blue-700" 
              : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
        )}>
            {icon}
            <span>{label}</span>
        </Link>
    )
}
